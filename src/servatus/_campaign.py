# pyright: reportPrivateUsage=false, reportUnnecessaryIsInstance=false

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import re
import stat
import tomllib
from collections.abc import Collection, Generator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import ClassVar, Self, cast

from . import _slurm
from ._errors import (
    AmbiguousSubmission,
    ConfigurationError,
    PlanError,
    ReconciliationError,
    SubmissionError,
    TaskConflict,
)

_SCHEMA_VERSION = 1
_PLAN_SCHEMA_VERSION = 1
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_GRES = re.compile(r"gpu(?::[A-Za-z0-9][A-Za-z0-9._-]*)?\Z")
_DURATION = re.compile(r"(?:(0|[1-9][0-9]*)-)?([0-9]{2}):([0-5][0-9]):([0-5][0-9])\Z")
_HEX_24 = re.compile(r"[0-9a-f]{24}\Z")
_HEX_32 = re.compile(r"[0-9a-f]{32}\Z")
_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_STATE_BYTES = 256 * 1024 * 1024


def _configuration(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)


def _integer(value: object, *, minimum: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}")
    return value


def _duration_seconds(value: object, *, name: str) -> int:
    if not isinstance(value, str) or (match := _DURATION.fullmatch(value)) is None:
        raise ConfigurationError(f"{name} must use canonical [days-]hours:minutes:seconds")
    days = int(match.group(1) or 0)
    hours = int(match.group(2))
    if match.group(1) is not None and hours > 23:
        raise ConfigurationError(f"{name} hours must be below 24 when days are present")
    seconds = ((days * 24 + hours) * 60 + int(match.group(3))) * 60 + int(match.group(4))
    if seconds <= 0:
        raise ConfigurationError(f"{name} must be positive, not unlimited")
    return seconds


def _effective_time_limit(value: str) -> str:
    seconds = _duration_seconds(value, name="time_limit")
    rounded = ((seconds + 59) // 60) * 60
    days, remainder = divmod(rounded, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes = remainder // 60
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:00"
    return f"{hours:02d}:{minutes:02d}:00"


def _safe_token(value: object, *, name: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise ConfigurationError(f"{name} must be one safe site token")
    return value


def _absolute_path(value: object, *, name: str) -> PurePosixPath:
    if not isinstance(value, (str, os.PathLike)):
        raise ConfigurationError(f"{name} must be an absolute POSIX path")
    raw = cast(str | bytes, os.fspath(value))
    if not isinstance(raw, str):
        raise ConfigurationError(f"{name} must be an absolute POSIX path")
    path = PurePosixPath(raw)
    if _CONTROL.search(raw) or not path.is_absolute() or ".." in path.parts or raw != str(path):
        raise ConfigurationError(f"{name} must be a normalized absolute POSIX path")
    return path


def _read_toml(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as source:
            document = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigurationError(f"cannot read TOML configuration: {path}") from error
    return cast(dict[str, object], document)


@dataclass(frozen=True, slots=True)
class Task:
    key: str
    args: tuple[str, ...]
    stdin: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ConfigurationError("Task.key must be a nonempty string")
        if not isinstance(self.args, tuple) or any(not isinstance(arg, str) for arg in self.args):
            raise ConfigurationError("Task.args must be a tuple of strings")
        if any("\0" in arg for arg in self.args):
            raise ConfigurationError("Task.args cannot contain NUL")
        if not isinstance(self.stdin, bytes):
            raise ConfigurationError("Task.stdin must be bytes")


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    cpus_per_task: int
    memory_mib_per_task: int
    gpus_per_task: int
    time_limit: str

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {"cpus_per_task", "memory_mib_per_task", "gpus_per_task", "time_limit"}
    )

    def __post_init__(self) -> None:
        _integer(self.cpus_per_task, minimum=1, name="cpus_per_task")
        _integer(self.memory_mib_per_task, minimum=1, name="memory_mib_per_task")
        _integer(self.gpus_per_task, minimum=0, name="gpus_per_task")
        _duration_seconds(self.time_limit, name="time_limit")

    @classmethod
    def from_toml(cls, path: Path) -> Self:
        values = _read_toml(path)
        unknown = values.keys() - cls._KEYS
        missing = cls._KEYS - values.keys()
        if unknown:
            raise ConfigurationError(f"unknown resource keys: {', '.join(sorted(unknown))}")
        if missing:
            raise ConfigurationError(f"missing resource keys: {', '.join(sorted(missing))}")
        return cls(
            cpus_per_task=_integer(values["cpus_per_task"], minimum=1, name="cpus_per_task"),
            memory_mib_per_task=_integer(
                values["memory_mib_per_task"], minimum=1, name="memory_mib_per_task"
            ),
            gpus_per_task=_integer(values["gpus_per_task"], minimum=0, name="gpus_per_task"),
            time_limit=cast(str, values["time_limit"]),
        )


@dataclass(frozen=True, slots=True)
class SlurmTarget:
    host: str
    slurm_bin: PurePosixPath
    apptainer: PurePosixPath
    image: PurePosixPath
    work_root: PurePosixPath
    log_root: PurePosixPath
    partitions: tuple[str, ...]
    account: str | None
    qos: str | None
    constraint: str | None
    gpu_gres: str | None
    max_tasks_per_allocation: int
    max_cpus_per_allocation: int
    max_memory_mib_per_allocation: int
    max_gpus_per_allocation: int
    max_time_limit: str
    max_allocations_per_submit: int
    max_script_bytes: int

    _REQUIRED: ClassVar[frozenset[str]] = frozenset(
        {
            "host",
            "slurm_bin",
            "apptainer",
            "image",
            "work_root",
            "log_root",
            "partitions",
            "max_tasks_per_allocation",
            "max_cpus_per_allocation",
            "max_memory_mib_per_allocation",
            "max_gpus_per_allocation",
            "max_time_limit",
            "max_allocations_per_submit",
            "max_script_bytes",
        }
    )
    _OPTIONAL: ClassVar[frozenset[str]] = frozenset({"account", "qos", "constraint", "gpu_gres"})

    def __post_init__(self) -> None:
        _safe_token(self.host, name="host")
        for name in ("slurm_bin", "apptainer", "image", "work_root", "log_root"):
            object.__setattr__(self, name, _absolute_path(getattr(self, name), name=name))
        _configuration(
            isinstance(self.partitions, tuple) and bool(self.partitions),
            "partitions must be a nonempty tuple",
        )
        for partition in self.partitions:
            _safe_token(partition, name="partition")
        _configuration(
            len(set(self.partitions)) == len(self.partitions), "partitions must be unique"
        )
        for name in ("account", "qos", "constraint"):
            _safe_token(getattr(self, name), name=name, optional=True)
        if self.gpu_gres is not None:
            _configuration(
                isinstance(self.gpu_gres, str) and _GRES.fullmatch(self.gpu_gres) is not None,
                "gpu_gres must be one count-free GPU resource",
            )
            suffix = self.gpu_gres.rsplit(":", 1)[-1]
            _configuration(
                not (":" in self.gpu_gres and suffix.isdecimal()),
                "gpu_gres cannot contain a count",
            )
        for name in (
            "max_tasks_per_allocation",
            "max_cpus_per_allocation",
            "max_memory_mib_per_allocation",
            "max_allocations_per_submit",
            "max_script_bytes",
        ):
            _integer(getattr(self, name), minimum=1, name=name)
        _integer(self.max_gpus_per_allocation, minimum=0, name="max_gpus_per_allocation")
        _duration_seconds(self.max_time_limit, name="max_time_limit")
        _configuration(
            (self.gpu_gres is None) == (self.max_gpus_per_allocation == 0),
            "gpu_gres and max_gpus_per_allocation conflict",
        )

    @classmethod
    def from_toml(cls, path: Path) -> Self:
        values = _read_toml(path)
        allowed = cls._REQUIRED | cls._OPTIONAL
        unknown = values.keys() - allowed
        missing = cls._REQUIRED - values.keys()
        if unknown:
            raise ConfigurationError(f"unknown target keys: {', '.join(sorted(unknown))}")
        if missing:
            raise ConfigurationError(f"missing target keys: {', '.join(sorted(missing))}")
        raw_partitions = values["partitions"]
        if not isinstance(raw_partitions, list):
            raise ConfigurationError("partitions must be an array of strings")
        partitions: list[str] = []
        for item in cast(list[object], raw_partitions):
            if not isinstance(item, str):
                raise ConfigurationError("partitions must be an array of strings")
            partitions.append(item)
        optional: dict[str, str | None] = {}
        for name in cls._OPTIONAL:
            value = values.get(name)
            if value is not None and not isinstance(value, str):
                raise ConfigurationError(f"{name} must be a string when present")
            optional[name] = value
        return cls(
            host=cast(str, values["host"]),
            slurm_bin=_absolute_path(values["slurm_bin"], name="slurm_bin"),
            apptainer=_absolute_path(values["apptainer"], name="apptainer"),
            image=_absolute_path(values["image"], name="image"),
            work_root=_absolute_path(values["work_root"], name="work_root"),
            log_root=_absolute_path(values["log_root"], name="log_root"),
            partitions=tuple(partitions),
            account=optional["account"],
            qos=optional["qos"],
            constraint=optional["constraint"],
            gpu_gres=optional["gpu_gres"],
            max_tasks_per_allocation=_integer(
                values["max_tasks_per_allocation"], minimum=1, name="max_tasks_per_allocation"
            ),
            max_cpus_per_allocation=_integer(
                values["max_cpus_per_allocation"], minimum=1, name="max_cpus_per_allocation"
            ),
            max_memory_mib_per_allocation=_integer(
                values["max_memory_mib_per_allocation"],
                minimum=1,
                name="max_memory_mib_per_allocation",
            ),
            max_gpus_per_allocation=_integer(
                values["max_gpus_per_allocation"], minimum=0, name="max_gpus_per_allocation"
            ),
            max_time_limit=cast(str, values["max_time_limit"]),
            max_allocations_per_submit=_integer(
                values["max_allocations_per_submit"],
                minimum=1,
                name="max_allocations_per_submit",
            ),
            max_script_bytes=_integer(
                values["max_script_bytes"], minimum=1, name="max_script_bytes"
            ),
        )


@dataclass(frozen=True, slots=True)
class PlannedAllocation:
    task_keys: tuple[str, ...]
    cpus: int
    memory_mib: int
    gpus: int
    time_limit: str


@dataclass(frozen=True, slots=True)
class _AllocationPlan:
    allocation_id: str
    allocation: PlannedAllocation
    script: bytes
    argv: tuple[str, ...]
    script_digest: str
    argv_digest: str


@dataclass(frozen=True, slots=True)
class SubmissionPlan:
    _campaign_id: str
    _state_revision: int
    _target: SlurmTarget
    _resources: ResourceRequest
    _completed: tuple[str, ...]
    _retry: tuple[str, ...]
    _tasks_per_allocation: int | None
    _allocations: tuple[_AllocationPlan, ...]
    _digest: str

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def allocations(self) -> tuple[PlannedAllocation, ...]:
        return tuple(item.allocation for item in self._allocations)


@dataclass(frozen=True, slots=True)
class JobReceipt:
    allocation_id: str
    job_id: int
    cluster: str | None
    task_keys: tuple[str, ...]

    def __str__(self) -> str:
        suffix = "" if self.cluster is None else f";{self.cluster}"
        return f"{self.job_id}{suffix}"


@dataclass(frozen=True, slots=True)
class CampaignStatus:
    pending_task_keys: tuple[str, ...]
    receipts: tuple[JobReceipt, ...]
    ambiguous_allocation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ValidationResult:
    shape: dict[str, object]
    shape_digest: str
    script_digest: str
    controller_stdout: str
    controller_stderr: str


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _task_record(task: Task) -> dict[str, object]:
    semantic = {
        "key": task.key,
        "args": list(task.args),
        "stdin": base64.b64encode(task.stdin).decode("ascii"),
    }
    return {**semantic, "digest": _digest(semantic)}


def _task_from_record(record: object) -> Task:
    if not isinstance(record, dict):
        raise TaskConflict("campaign task record is invalid")
    mapping = cast(dict[str, object], record)
    if set(mapping) != {"key", "args", "stdin", "digest"}:
        raise TaskConflict("campaign task record is invalid")
    try:
        task = Task(
            cast(str, mapping["key"]),
            tuple(cast(list[str], mapping["args"])),
            base64.b64decode(cast(str, mapping["stdin"]), validate=True),
        )
    except (TypeError, ValueError, ConfigurationError) as error:
        raise TaskConflict("campaign task record is invalid") from error
    if _task_record(task) != mapping:
        raise TaskConflict("campaign task record digest is invalid")
    return task


def _resource_dict(resources: ResourceRequest) -> dict[str, object]:
    return {
        "cpus_per_task": resources.cpus_per_task,
        "memory_mib_per_task": resources.memory_mib_per_task,
        "gpus_per_task": resources.gpus_per_task,
        "time_limit": resources.time_limit,
    }


def _target_dict(target: SlurmTarget) -> dict[str, object]:
    return {
        "host": target.host,
        "slurm_bin": str(target.slurm_bin),
        "apptainer": str(target.apptainer),
        "image": str(target.image),
        "work_root": str(target.work_root),
        "log_root": str(target.log_root),
        "partitions": list(target.partitions),
        "account": target.account,
        "qos": target.qos,
        "constraint": target.constraint,
        "gpu_gres": target.gpu_gres,
        "max_tasks_per_allocation": target.max_tasks_per_allocation,
        "max_cpus_per_allocation": target.max_cpus_per_allocation,
        "max_memory_mib_per_allocation": target.max_memory_mib_per_allocation,
        "max_gpus_per_allocation": target.max_gpus_per_allocation,
        "max_time_limit": target.max_time_limit,
        "max_allocations_per_submit": target.max_allocations_per_submit,
        "max_script_bytes": target.max_script_bytes,
    }


def _target_from_dict(value: object) -> SlurmTarget:
    if not isinstance(value, dict):
        raise PlanError("plan target is invalid")
    mapping = cast(dict[str, object], value)
    if frozenset(mapping) != SlurmTarget._REQUIRED | SlurmTarget._OPTIONAL:
        raise PlanError("plan target keys are invalid")
    try:
        partitions = mapping["partitions"]
        if not isinstance(partitions, list):
            raise TypeError
        target = SlurmTarget(
            host=cast(str, mapping["host"]),
            slurm_bin=PurePosixPath(cast(str, mapping["slurm_bin"])),
            apptainer=PurePosixPath(cast(str, mapping["apptainer"])),
            image=PurePosixPath(cast(str, mapping["image"])),
            work_root=PurePosixPath(cast(str, mapping["work_root"])),
            log_root=PurePosixPath(cast(str, mapping["log_root"])),
            partitions=tuple(cast(list[str], partitions)),
            account=cast(str | None, mapping["account"]),
            qos=cast(str | None, mapping["qos"]),
            constraint=cast(str | None, mapping["constraint"]),
            gpu_gres=cast(str | None, mapping["gpu_gres"]),
            max_tasks_per_allocation=cast(int, mapping["max_tasks_per_allocation"]),
            max_cpus_per_allocation=cast(int, mapping["max_cpus_per_allocation"]),
            max_memory_mib_per_allocation=cast(int, mapping["max_memory_mib_per_allocation"]),
            max_gpus_per_allocation=cast(int, mapping["max_gpus_per_allocation"]),
            max_time_limit=cast(str, mapping["max_time_limit"]),
            max_allocations_per_submit=cast(int, mapping["max_allocations_per_submit"]),
            max_script_bytes=cast(int, mapping["max_script_bytes"]),
        )
    except (KeyError, TypeError, ConfigurationError) as error:
        raise PlanError("plan target is invalid") from error
    if _target_dict(target) != mapping:
        raise PlanError("plan target types are invalid")
    return target


def _resource_from_dict(value: object) -> ResourceRequest:
    if not isinstance(value, dict):
        raise PlanError("plan resources are invalid")
    mapping = cast(dict[str, object], value)
    if frozenset(mapping) != ResourceRequest._KEYS:
        raise PlanError("plan resource keys are invalid")
    try:
        resources = ResourceRequest(
            cpus_per_task=cast(int, mapping["cpus_per_task"]),
            memory_mib_per_task=cast(int, mapping["memory_mib_per_task"]),
            gpus_per_task=cast(int, mapping["gpus_per_task"]),
            time_limit=cast(str, mapping["time_limit"]),
        )
    except (KeyError, ConfigurationError) as error:
        raise PlanError("plan resources are invalid") from error
    if _resource_dict(resources) != mapping:
        raise PlanError("plan resource types are invalid")
    return resources


def _allocation_summary(item: _AllocationPlan) -> dict[str, object]:
    allocation = item.allocation
    return {
        "allocation_id": item.allocation_id,
        "task_keys": list(allocation.task_keys),
        "cpus": allocation.cpus,
        "memory_mib": allocation.memory_mib,
        "gpus": allocation.gpus,
        "time_limit": allocation.time_limit,
        "sbatch_argv": list(item.argv),
        "script_digest": item.script_digest,
        "argv_digest": item.argv_digest,
    }


def _lineage(target: SlurmTarget, resources: ResourceRequest) -> dict[str, object]:
    target_values = _target_dict(target)
    resource_values = _resource_dict(resources)
    return {
        "target": target_values,
        "resources": resource_values,
        "target_digest": _digest(target_values),
        "resource_digest": _digest(resource_values),
    }


def _plan_semantics(plan: SubmissionPlan) -> dict[str, object]:
    return {
        "campaign_id": plan._campaign_id,
        "state_revision": plan._state_revision,
        "target": _target_dict(plan._target),
        "resources": _resource_dict(plan._resources),
        "completed": list(plan._completed),
        "retry": list(plan._retry),
        "tasks_per_allocation": plan._tasks_per_allocation,
        "allocations": [_allocation_summary(item) for item in plan._allocations],
    }


def plan_document(plan: SubmissionPlan) -> dict[str, object]:
    return {
        "schema_version": _PLAN_SCHEMA_VERSION,
        **_plan_semantics(plan),
        "digest": plan.digest,
    }


def sensitive_script_document(plan: SubmissionPlan) -> dict[str, object]:
    return {
        "sensitive": True,
        "plan_digest": plan.digest,
        "scripts": [
            {
                "allocation_id": allocation.allocation_id,
                "script": allocation.script.decode("utf-8"),
            }
            for allocation in plan._allocations
        ],
    }


class Campaign:
    def __init__(self, path: Path, tasks: tuple[Task, ...], entry: os.stat_result) -> None:
        self._path = path
        self._tasks = tasks
        self._entry = entry

    @classmethod
    def open(cls, path: Path, tasks: Sequence[Task]) -> Self:
        frozen = tuple(tasks)
        if any(not isinstance(task, Task) for task in frozen):
            raise ConfigurationError("tasks must contain only Task values")
        keys = [task.key for task in frozen]
        if len(set(keys)) != len(keys):
            raise ConfigurationError("Task keys must be unique")
        campaign_path, entry = _open_or_create_directory(path)
        campaign = cls(campaign_path, frozen, entry)
        with campaign._locked_state(create=True) as state:
            if state is None:
                campaign._write_state(
                    {
                        "schema_version": _SCHEMA_VERSION,
                        "campaign_id": os.urandom(16).hex(),
                        "revision": 0,
                        "tasks": [_task_record(task) for task in frozen],
                        "lineage": None,
                        "intents": [],
                        "receipts": [],
                        "resolutions": [],
                    }
                )
            else:
                stored = tuple(
                    _task_from_record(record) for record in cast(list[object], state["tasks"])
                )
                if stored != frozen:
                    raise TaskConflict("campaign was reopened with changed or reordered tasks")
        return campaign

    @classmethod
    def _reopen(cls, path: Path) -> Self:
        campaign_path, entry = _open_existing_directory(path)
        temporary = cls(campaign_path, (), entry)
        state = temporary._read_state()
        tasks = tuple(_task_from_record(record) for record in cast(list[object], state["tasks"]))
        return cls(campaign_path, tasks, entry)

    def plan(
        self,
        target: SlurmTarget,
        resources: ResourceRequest,
        *,
        completed: Collection[str] = (),
        retry: Collection[str] = (),
        tasks_per_allocation: int | None = None,
    ) -> SubmissionPlan:
        if not isinstance(target, SlurmTarget) or not isinstance(resources, ResourceRequest):
            raise ConfigurationError("plan requires a SlurmTarget and ResourceRequest")
        state = self._read_state()
        ambiguous = _ambiguous_ids(state)
        if ambiguous:
            raise AmbiguousSubmission("resolve ambiguous allocation intent before planning")
        known = tuple(task.key for task in self._tasks)
        completed_values = _ordered_selection(completed, known, "completed")
        retry_values = _ordered_selection(retry, known, "retry")
        if set(completed_values) & set(retry_values):
            raise PlanError("completed and retry selections overlap")
        receipts = _receipt_values(state)
        accepted = {key for receipt in receipts for key in receipt.task_keys}
        if any(key not in accepted for key in retry_values):
            raise PlanError("retry requires an earlier accepted receipt")
        lineage = state["lineage"]
        if lineage is not None and lineage != _lineage(target, resources):
            raise PlanError("campaign is bound to different target or resource semantics")
        capacity = _capacity(target, resources)
        if tasks_per_allocation is not None:
            requested = _integer(tasks_per_allocation, minimum=1, name="tasks_per_allocation")
            if requested > capacity:
                raise PlanError("tasks_per_allocation exceeds feasible capacity")
            capacity = requested
        selected = tuple(
            task
            for task in self._tasks
            if task.key not in completed_values
            and (task.key not in accepted or task.key in retry_values)
        )
        groups = _balanced_groups(selected, capacity)
        campaign_id = cast(str, state["campaign_id"])
        revision = cast(int, state["revision"])
        seed = _digest(
            {
                "campaign_id": campaign_id,
                "revision": revision,
                "target": _target_dict(target),
                "resources": _resource_dict(resources),
                "completed": list(completed_values),
                "retry": list(retry_values),
                "tasks_per_allocation": tasks_per_allocation,
                "groups": [[task.key for task in group] for group in groups],
            }
        )
        allocations: list[_AllocationPlan] = []
        effective_time_limit = _effective_time_limit(resources.time_limit)
        for index, group in enumerate(groups):
            allocation_id = hashlib.sha256(f"{seed}:{index}".encode()).hexdigest()[:24]
            script = _slurm.render_script(target, resources, group, allocation_id)
            if len(script) > target.max_script_bytes:
                raise PlanError(
                    f"rendered script is {len(script)} bytes; target permits "
                    f"{target.max_script_bytes}"
                )
            argv = _slurm.sbatch_argv(
                target,
                resources,
                len(group),
                allocation_id,
                effective_time_limit,
            )
            public = PlannedAllocation(
                task_keys=tuple(task.key for task in group),
                cpus=len(group) * resources.cpus_per_task,
                memory_mib=len(group) * resources.memory_mib_per_task,
                gpus=len(group) * resources.gpus_per_task,
                time_limit=effective_time_limit,
            )
            allocations.append(
                _AllocationPlan(
                    allocation_id,
                    public,
                    script,
                    argv,
                    hashlib.sha256(script).hexdigest(),
                    _digest(list(argv)),
                )
            )
        partial = SubmissionPlan(
            campaign_id,
            revision,
            target,
            resources,
            completed_values,
            retry_values,
            tasks_per_allocation,
            tuple(allocations),
            "",
        )
        return SubmissionPlan(
            campaign_id,
            revision,
            target,
            resources,
            completed_values,
            retry_values,
            tasks_per_allocation,
            tuple(allocations),
            _digest(_plan_semantics(partial)),
        )

    def submit(self, plan: SubmissionPlan) -> tuple[JobReceipt, ...]:
        self._verify_plan(plan)
        expected_revision = plan._state_revision
        receipts: list[JobReceipt] = []
        for allocation in plan._allocations[: plan._target.max_allocations_per_submit]:
            expected_revision = self._record_intent(
                plan, allocation, expected_revision=expected_revision
            )
            try:
                result = _slurm._run_ssh(plan._target, allocation.argv, allocation.script)
            except BaseException as error:
                raise AmbiguousSubmission(
                    f"allocation {allocation.allocation_id} may have been accepted"
                ) from error
            if result.returncode != 0:
                raise AmbiguousSubmission(
                    f"allocation {allocation.allocation_id} has no provable acceptance receipt"
                )
            try:
                job_id, cluster = _slurm.parse_receipt(result.stdout)
            except ValueError as error:
                raise AmbiguousSubmission(
                    f"allocation {allocation.allocation_id} returned an invalid receipt"
                ) from error
            receipt = JobReceipt(
                allocation.allocation_id,
                job_id,
                cluster,
                allocation.allocation.task_keys,
            )
            try:
                expected_revision = self._record_receipt(
                    receipt, expected_revision=expected_revision
                )
            except BaseException as error:
                raise AmbiguousSubmission(
                    f"allocation {allocation.allocation_id} was accepted but receipt sync failed"
                ) from error
            receipts.append(receipt)
        return tuple(receipts)

    def reconcile(self, target: SlurmTarget, allocation_id: str) -> JobReceipt:
        state = self._read_state()
        intent = _unresolved_intent(state, allocation_id)
        if intent["target_digest"] != _digest(_target_dict(target)):
            raise ReconciliationError("reconciliation target does not match allocation intent")
        match = _slurm.query_identity(
            target,
            job_name=cast(str, intent["job_name"]),
            window_start=cast(str, intent["window_start"]),
            window_end=cast(str, intent["window_end"]),
        )
        receipt = JobReceipt(
            allocation_id,
            match.job_id,
            match.cluster,
            tuple(cast(list[str], intent["task_keys"])),
        )
        self._record_receipt(receipt, expected_revision=cast(int, state["revision"]))
        return receipt

    def resolve(
        self,
        allocation_id: str,
        *,
        job_id: int | None,
        cluster: str | None = None,
    ) -> None:
        if job_id is not None:
            _integer(job_id, minimum=1, name="job_id")
        if cluster is not None:
            _safe_token(cluster, name="cluster")
        if job_id is None and cluster is not None:
            raise ConfigurationError("cluster requires a job_id")
        with self._locked_state() as state:
            assert state is not None
            intent = _unresolved_intent(state, allocation_id)
            if job_id is None:
                cast(list[dict[str, object]], state["resolutions"]).append(
                    {"allocation_id": allocation_id, "accepted": False}
                )
            else:
                cast(list[dict[str, object]], state["receipts"]).append(
                    {
                        "allocation_id": allocation_id,
                        "job_id": job_id,
                        "cluster": cluster,
                        "task_keys": intent["task_keys"],
                    }
                )
            state["revision"] = cast(int, state["revision"]) + 1
            self._write_state(state)

    def status(self) -> CampaignStatus:
        state = self._read_state()
        receipts = _receipt_values(state)
        accepted = {key for receipt in receipts for key in receipt.task_keys}
        return CampaignStatus(
            tuple(task.key for task in self._tasks if task.key not in accepted),
            receipts,
            tuple(_ambiguous_ids(state)),
        )

    def _verify_plan(self, plan: SubmissionPlan) -> None:
        if not isinstance(plan, SubmissionPlan):
            raise PlanError("submit requires a SubmissionPlan")
        state = self._read_state()
        if plan._campaign_id != state["campaign_id"]:
            raise PlanError("submission plan belongs to another campaign")
        if plan._state_revision != state["revision"]:
            raise PlanError("submission plan is stale")
        for allocation in plan._allocations:
            if allocation.script_digest != hashlib.sha256(allocation.script).hexdigest():
                raise PlanError("submission plan script was changed")
            if allocation.argv_digest != _digest(list(allocation.argv)):
                raise PlanError("submission plan command was changed")
        if plan.digest != _digest(_plan_semantics(plan)):
            raise PlanError("submission plan was changed")
        if _ambiguous_ids(state):
            raise AmbiguousSubmission("resolve ambiguous allocation intent before submission")
        lineage = state["lineage"]
        expected = _lineage(plan._target, plan._resources)
        if lineage is not None and lineage != expected:
            raise PlanError("campaign is bound to different target or resource semantics")

    def _record_intent(
        self,
        plan: SubmissionPlan,
        allocation: _AllocationPlan,
        *,
        expected_revision: int,
    ) -> int:
        now = datetime.now(UTC)
        with self._locked_state() as state:
            assert state is not None
            if state["revision"] != expected_revision or _ambiguous_ids(state):
                raise PlanError("campaign changed before submission intent")
            lineage = _lineage(plan._target, plan._resources)
            if state["lineage"] is None:
                state["lineage"] = lineage
            elif state["lineage"] != lineage:
                raise PlanError("campaign is bound to different target or resource semantics")
            cast(list[dict[str, object]], state["intents"]).append(
                {
                    "allocation_id": allocation.allocation_id,
                    "task_keys": list(allocation.allocation.task_keys),
                    "plan_digest": plan.digest,
                    "script_digest": allocation.script_digest,
                    "argv_digest": allocation.argv_digest,
                    "target_digest": lineage["target_digest"],
                    "resource_digest": lineage["resource_digest"],
                    "allocation": {
                        "cpus": allocation.allocation.cpus,
                        "memory_mib": allocation.allocation.memory_mib,
                        "gpus": allocation.allocation.gpus,
                        "time_limit": allocation.allocation.time_limit,
                    },
                    "sbatch_argv": list(allocation.argv),
                    "job_name": f"servatus-{allocation.allocation_id}",
                    "window_start": (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S"),
                    "window_end": (now + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S"),
                }
            )
            state["revision"] = expected_revision + 1
            self._write_state(state)
            return expected_revision + 1

    def _record_receipt(self, receipt: JobReceipt, *, expected_revision: int) -> int:
        with self._locked_state() as state:
            assert state is not None
            if state["revision"] != expected_revision:
                raise PlanError("campaign changed before receipt")
            _unresolved_intent(state, receipt.allocation_id)
            cast(list[dict[str, object]], state["receipts"]).append(
                {
                    "allocation_id": receipt.allocation_id,
                    "job_id": receipt.job_id,
                    "cluster": receipt.cluster,
                    "task_keys": list(receipt.task_keys),
                }
            )
            state["revision"] = expected_revision + 1
            self._write_state(state)
            return expected_revision + 1

    def _read_state(self) -> dict[str, object]:
        with self._locked_state() as state:
            assert state is not None
            return state

    @contextmanager
    def _locked_state(self, *, create: bool = False) -> Generator[dict[str, object] | None]:
        descriptor = _open_campaign_directory(self._path, self._entry)
        lock_fd = -1
        try:
            lock_fd = os.open(
                ".lock",
                os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=descriptor,
            )
            _require_owner_file(os.fstat(lock_fd), ".lock")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                state = _load_state(descriptor)
            except FileNotFoundError:
                if not create:
                    raise TaskConflict("campaign state is missing") from None
                state = None
            self._active_descriptor = descriptor
            try:
                yield state
            finally:
                del self._active_descriptor
        finally:
            if lock_fd >= 0:
                os.close(lock_fd)
            os.close(descriptor)

    def _write_state(self, state: dict[str, object]) -> None:
        descriptor = self._active_descriptor
        _validate_state(state)
        encoded = _canonical(state) + b"\n"
        if len(encoded) > _MAX_STATE_BYTES:
            raise TaskConflict(
                f"campaign state is {len(encoded)} bytes; maximum is {_MAX_STATE_BYTES}"
            )
        name = f".campaign-{os.urandom(12).hex()}.tmp"
        stage = -1
        installed = False
        try:
            stage = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=descriptor,
            )
            view = memoryview(encoded)
            while view:
                written = os.write(stage, view)
                if written <= 0:
                    raise OSError("campaign state write made no progress")
                view = view[written:]
            os.fsync(stage)
            os.close(stage)
            stage = -1
            try:
                current = os.stat("campaign.json", dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                current = None
            if current is not None:
                _require_owner_file(current, "campaign.json")
            os.replace(name, "campaign.json", src_dir_fd=descriptor, dst_dir_fd=descriptor)
            installed = True
            os.fsync(descriptor)
        except BaseException:
            if stage >= 0:
                with suppress(OSError):
                    os.close(stage)
            if not installed:
                with suppress(FileNotFoundError):
                    os.unlink(name, dir_fd=descriptor)
            raise


def _ordered_selection(
    values: Collection[str], known: tuple[str, ...], name: str
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or any(not isinstance(value, str) for value in values):
        raise PlanError(f"{name} must be a collection of task keys")
    selected = set(values)
    unknown = selected - set(known)
    if unknown:
        raise PlanError(f"{name} contains unknown task keys")
    return tuple(key for key in known if key in selected)


def _capacity(target: SlurmTarget, resources: ResourceRequest) -> int:
    if _duration_seconds(_effective_time_limit(resources.time_limit), name="time_limit") > (
        _duration_seconds(_effective_time_limit(target.max_time_limit), name="max_time_limit")
    ):
        raise PlanError("time_limit exceeds target ceiling")
    capacities = [
        target.max_tasks_per_allocation,
        target.max_cpus_per_allocation // resources.cpus_per_task,
        target.max_memory_mib_per_allocation // resources.memory_mib_per_task,
    ]
    if resources.gpus_per_task:
        if target.gpu_gres is None:
            raise PlanError("GPU work requires a target GPU GRES")
        capacities.append(target.max_gpus_per_allocation // resources.gpus_per_task)
    capacity = min(capacities)
    if capacity < 1:
        raise PlanError("one task exceeds target capacity")
    return capacity


def _balanced_groups(tasks: tuple[Task, ...], capacity: int) -> tuple[tuple[Task, ...], ...]:
    if not tasks:
        return ()
    count = len(tasks)
    group_count = (count + capacity - 1) // capacity
    small, larger = divmod(count, group_count)
    sizes = [small + 1] * larger + [small] * (group_count - larger)
    groups: list[tuple[Task, ...]] = []
    offset = 0
    for size in sizes:
        groups.append(tasks[offset : offset + size])
        offset += size
    return tuple(groups)


def _receipt_values(state: dict[str, object]) -> tuple[JobReceipt, ...]:
    values: list[JobReceipt] = []
    for raw in cast(list[dict[str, object]], state["receipts"]):
        values.append(
            JobReceipt(
                cast(str, raw["allocation_id"]),
                cast(int, raw["job_id"]),
                cast(str | None, raw["cluster"]),
                tuple(cast(list[str], raw["task_keys"])),
            )
        )
    return tuple(values)


def _ambiguous_ids(state: dict[str, object]) -> list[str]:
    resolved = {
        cast(str, value["allocation_id"])
        for value in cast(list[dict[str, object]], state["resolutions"])
    }
    receipted = {receipt.allocation_id for receipt in _receipt_values(state)}
    return [
        cast(str, intent["allocation_id"])
        for intent in cast(list[dict[str, object]], state["intents"])
        if intent["allocation_id"] not in receipted and intent["allocation_id"] not in resolved
    ]


def _unresolved_intent(state: dict[str, object], allocation_id: str) -> dict[str, object]:
    if allocation_id not in _ambiguous_ids(state):
        raise ReconciliationError("allocation is not an unresolved ambiguous intent")
    for intent in cast(list[dict[str, object]], state["intents"]):
        if intent["allocation_id"] == allocation_id:
            return intent
    raise ReconciliationError("allocation intent is missing")


def _open_or_create_directory(path: Path) -> tuple[Path, os.stat_result]:
    raw = os.fspath(path)
    if "\0" in raw:
        raise TaskConflict("campaign path contains NUL")
    normalized = Path(os.path.abspath(raw))
    name = normalized.name
    if name in {"", ".", ".."}:
        raise TaskConflict("campaign path must name one directory")
    try:
        parent_fd = os.open(
            normalized.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError as error:
        raise TaskConflict("campaign parent is unavailable or unsafe") from error
    child_fd = -1
    created = False
    try:
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
            created = True
        except FileExistsError:
            pass
        try:
            child_fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
        except OSError as error:
            raise TaskConflict("campaign directory is unavailable or unsafe") from error
        entry = os.fstat(child_fd)
        _require_owner_directory(entry)
        os.fsync(parent_fd)
        return normalized, entry
    except BaseException as error:
        if created:
            try:
                os.rmdir(name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError as cleanup_error:
                error.add_note(
                    f"Servatus could not remove unsynced campaign directory: {cleanup_error}"
                )
        raise
    finally:
        if child_fd >= 0:
            os.close(child_fd)
        os.close(parent_fd)


def _open_existing_directory(path: Path) -> tuple[Path, os.stat_result]:
    raw = os.fspath(path)
    if "\0" in raw:
        raise TaskConflict("campaign path contains NUL")
    normalized = Path(os.path.abspath(raw))
    try:
        descriptor = os.open(
            normalized, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
    except OSError as error:
        raise TaskConflict("campaign directory is unavailable or unsafe") from error
    try:
        entry = os.fstat(descriptor)
        _require_owner_directory(entry)
    finally:
        os.close(descriptor)
    return normalized, entry


def _open_campaign_directory(path: Path, expected: os.stat_result) -> int:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    entry = os.fstat(descriptor)
    try:
        _require_owner_directory(entry)
        if entry.st_dev != expected.st_dev or entry.st_ino != expected.st_ino:
            raise TaskConflict("campaign directory was replaced")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _require_owner_directory(entry: os.stat_result) -> None:
    if not stat.S_ISDIR(entry.st_mode) or entry.st_uid != os.getuid() or entry.st_mode & 0o077:
        raise TaskConflict("campaign directory must be owner-only and symlink-free")


def _require_owner_file(entry: os.stat_result, name: str) -> None:
    if not stat.S_ISREG(entry.st_mode) or entry.st_uid != os.getuid() or entry.st_mode & 0o077:
        raise TaskConflict(f"campaign {name} must be an owner-only regular file")


def _load_state(descriptor: int) -> dict[str, object]:
    state_fd = os.open(
        "campaign.json",
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        dir_fd=descriptor,
    )
    try:
        entry = os.fstat(state_fd)
        _require_owner_file(entry, "campaign.json")
        if entry.st_size > _MAX_STATE_BYTES:
            raise TaskConflict("campaign state is too large")
        chunks: list[bytes] = []
        remaining = entry.st_size + 1
        while remaining and (chunk := os.read(state_fd, min(remaining, 1024 * 1024))):
            chunks.append(chunk)
            remaining -= len(chunk)
        if sum(map(len, chunks)) != entry.st_size:
            raise TaskConflict("campaign state changed while reading")
    finally:
        os.close(state_fd)
    try:
        state = json.loads(b"".join(chunks))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TaskConflict("campaign state is invalid JSON") from error
    if not isinstance(state, dict):
        raise TaskConflict("campaign state is invalid")
    typed = cast(dict[str, object], state)
    _validate_state(typed)
    return typed


def _validate_state(state: dict[str, object]) -> None:
    keys = {
        "schema_version",
        "campaign_id",
        "revision",
        "tasks",
        "lineage",
        "intents",
        "receipts",
        "resolutions",
    }
    if set(state) != keys or state.get("schema_version") != _SCHEMA_VERSION:
        raise TaskConflict("campaign state schema is unsupported")
    if (
        not isinstance(state["campaign_id"], str)
        or _HEX_32.fullmatch(state["campaign_id"]) is None
        or not isinstance(state["revision"], int)
        or isinstance(state["revision"], bool)
        or state["revision"] < 0
        or not isinstance(state["tasks"], list)
        or not isinstance(state["intents"], list)
        or not isinstance(state["receipts"], list)
        or not isinstance(state["resolutions"], list)
    ):
        raise TaskConflict("campaign state values are invalid")
    for record in cast(list[object], state["tasks"]):
        _task_from_record(record)
    task_keys = tuple(
        _task_from_record(record).key for record in cast(list[object], state["tasks"])
    )
    if len(set(task_keys)) != len(task_keys):
        raise TaskConflict("campaign task keys are not unique")
    lineage = _validate_lineage(state["lineage"])
    intents = _validate_intents(cast(list[object], state["intents"]), task_keys, lineage)
    _validate_receipts(cast(list[object], state["receipts"]), intents)
    _validate_resolutions(cast(list[object], state["resolutions"]), intents)


def _validate_lineage(
    value: object,
) -> tuple[SlurmTarget, ResourceRequest] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TaskConflict("campaign lineage is invalid")
    lineage = cast(dict[str, object], value)
    if set(lineage) != {"target", "resources", "target_digest", "resource_digest"}:
        raise TaskConflict("campaign lineage is invalid")
    try:
        target = _target_from_dict(lineage["target"])
        resources = _resource_from_dict(lineage["resources"])
    except PlanError as error:
        raise TaskConflict("campaign lineage is invalid") from error
    if (
        lineage["target"] != _target_dict(target)
        or lineage["resources"] != _resource_dict(resources)
        or lineage["target_digest"] != _digest(_target_dict(target))
        or lineage["resource_digest"] != _digest(_resource_dict(resources))
    ):
        raise TaskConflict("campaign lineage is invalid")
    return target, resources


def _validate_intents(
    value: object,
    task_keys: tuple[str, ...],
    lineage: tuple[SlurmTarget, ResourceRequest] | None,
) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, list):
        raise TaskConflict("campaign intents are invalid")
    expected = {
        "allocation_id",
        "task_keys",
        "plan_digest",
        "script_digest",
        "argv_digest",
        "target_digest",
        "resource_digest",
        "allocation",
        "sbatch_argv",
        "job_name",
        "window_start",
        "window_end",
    }
    intents: dict[str, tuple[str, ...]] = {}
    for raw in cast(list[object], value):
        if not isinstance(raw, dict):
            raise TaskConflict("campaign intent is invalid")
        intent = cast(dict[str, object], raw)
        allocation_id = intent.get("allocation_id")
        raw_keys = intent.get("task_keys")
        if (
            set(intent) != expected
            or not isinstance(allocation_id, str)
            or _HEX_24.fullmatch(allocation_id) is None
            or allocation_id in intents
            or not isinstance(raw_keys, list)
        ):
            raise TaskConflict("campaign intent is invalid")
        intent_keys = tuple(cast(list[object], raw_keys))
        if (
            not intent_keys
            or any(not isinstance(key, str) or key not in task_keys for key in intent_keys)
            or len(set(intent_keys)) != len(intent_keys)
        ):
            raise TaskConflict("campaign intent task keys are invalid")
        if lineage is None:
            raise TaskConflict("campaign intent has no resource lineage")
        target, resources = lineage
        for name in (
            "plan_digest",
            "script_digest",
            "argv_digest",
            "target_digest",
            "resource_digest",
        ):
            digest = intent[name]
            if not isinstance(digest, str) or _HEX_64.fullmatch(digest) is None:
                raise TaskConflict("campaign intent digest is invalid")
        if intent["target_digest"] != _digest(_target_dict(target)) or intent[
            "resource_digest"
        ] != _digest(_resource_dict(resources)):
            raise TaskConflict("campaign intent lineage is invalid")
        allocation = intent["allocation"]
        argv = intent["sbatch_argv"]
        if not isinstance(allocation, dict) or not isinstance(argv, list):
            raise TaskConflict("campaign intent provenance is invalid")
        totals = cast(dict[str, object], allocation)
        typed_argv = cast(list[object], argv)
        task_count = len(intent_keys)
        expected_totals = {
            "cpus": task_count * resources.cpus_per_task,
            "memory_mib": task_count * resources.memory_mib_per_task,
            "gpus": task_count * resources.gpus_per_task,
            "time_limit": _effective_time_limit(resources.time_limit),
        }
        expected_argv = list(
            _slurm.sbatch_argv(
                target,
                resources,
                task_count,
                allocation_id,
                _effective_time_limit(resources.time_limit),
            )
        )
        if (
            totals != expected_totals
            or not typed_argv
            or any(not isinstance(argument, str) or "\0" in argument for argument in typed_argv)
            or typed_argv != expected_argv
            or intent["argv_digest"] != _digest(typed_argv)
        ):
            raise TaskConflict("campaign intent provenance is invalid")
        if intent["job_name"] != f"servatus-{allocation_id}" or any(
            not isinstance(intent[name], str) or _CONTROL.search(cast(str, intent[name]))
            for name in ("window_start", "window_end")
        ):
            raise TaskConflict("campaign intent identity is invalid")
        intents[allocation_id] = cast(tuple[str, ...], intent_keys)
    return intents


def _validate_receipts(value: object, intents: dict[str, tuple[str, ...]]) -> None:
    if not isinstance(value, list):
        raise TaskConflict("campaign receipts are invalid")
    seen: set[str] = set()
    for raw in cast(list[object], value):
        if not isinstance(raw, dict):
            raise TaskConflict("campaign receipt is invalid")
        receipt = cast(dict[str, object], raw)
        allocation_id = receipt.get("allocation_id")
        job_id = receipt.get("job_id")
        cluster = receipt.get("cluster")
        task_keys = receipt.get("task_keys")
        if (
            set(receipt) != {"allocation_id", "job_id", "cluster", "task_keys"}
            or not isinstance(allocation_id, str)
            or allocation_id not in intents
            or allocation_id in seen
            or isinstance(job_id, bool)
            or not isinstance(job_id, int)
            or job_id <= 0
            or (
                cluster is not None
                and (not isinstance(cluster, str) or _TOKEN.fullmatch(cluster) is None)
            )
            or not isinstance(task_keys, list)
            or tuple(cast(list[object], task_keys)) != intents[allocation_id]
        ):
            raise TaskConflict("campaign receipt is invalid")
        seen.add(allocation_id)


def _validate_resolutions(value: object, intents: dict[str, tuple[str, ...]]) -> None:
    if not isinstance(value, list):
        raise TaskConflict("campaign resolutions are invalid")
    seen: set[str] = set()
    for raw in cast(list[object], value):
        if not isinstance(raw, dict):
            raise TaskConflict("campaign resolution is invalid")
        resolution = cast(dict[str, object], raw)
        allocation_id = resolution.get("allocation_id")
        if (
            set(resolution) != {"allocation_id", "accepted"}
            or not isinstance(allocation_id, str)
            or allocation_id not in intents
            or allocation_id in seen
            or resolution["accepted"] is not False
        ):
            raise TaskConflict("campaign resolution is invalid")
        seen.add(allocation_id)


def _plan_integer(value: object, *, minimum: int, name: str) -> int:
    if type(value) is not int or value < minimum:
        raise PlanError(f"plan {name} must be an integer >= {minimum}")
    return value


def _plan_strings(value: object, *, name: str, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise PlanError(f"plan {name} must be an array of strings")
    strings: list[str] = []
    for item in cast(list[object], value):
        if not isinstance(item, str) or "\0" in item:
            raise PlanError(f"plan {name} must be an array of strings")
        strings.append(item)
    if nonempty and not strings:
        raise PlanError(f"plan {name} cannot be empty")
    return strings


def _validate_plan_document(document: object) -> dict[str, object]:
    if not isinstance(document, dict):
        raise PlanError("plan document must be an object")
    mapping = cast(dict[str, object], document)
    expected = {
        "schema_version",
        "campaign_id",
        "state_revision",
        "target",
        "resources",
        "completed",
        "retry",
        "tasks_per_allocation",
        "allocations",
        "digest",
    }
    if set(mapping) != expected:
        raise PlanError("plan document keys are invalid")
    if (
        type(mapping["schema_version"]) is not int
        or mapping["schema_version"] != _PLAN_SCHEMA_VERSION
    ):
        raise PlanError("plan document schema is unsupported")
    campaign_id = mapping["campaign_id"]
    digest = mapping["digest"]
    if not isinstance(campaign_id, str) or _HEX_32.fullmatch(campaign_id) is None:
        raise PlanError("plan campaign identity is invalid")
    _plan_integer(mapping["state_revision"], minimum=0, name="state_revision")
    _target_from_dict(mapping["target"])
    _resource_from_dict(mapping["resources"])
    completed = _plan_strings(mapping["completed"], name="completed")
    retry = _plan_strings(mapping["retry"], name="retry")
    if len(set(completed)) != len(completed) or len(set(retry)) != len(retry):
        raise PlanError("plan selections contain duplicate task keys")
    tasks_per = mapping["tasks_per_allocation"]
    if tasks_per is not None:
        _plan_integer(tasks_per, minimum=1, name="tasks_per_allocation")
    allocations = mapping["allocations"]
    if not isinstance(allocations, list):
        raise PlanError("plan allocations must be an array")
    for raw in cast(list[object], allocations):
        _validate_plan_allocation(raw)
    if not isinstance(digest, str) or _HEX_64.fullmatch(digest) is None:
        raise PlanError("plan digest is invalid")
    return mapping


def _validate_plan_allocation(value: object) -> None:
    if not isinstance(value, dict):
        raise PlanError("plan allocation must be an object")
    allocation = cast(dict[str, object], value)
    expected = {
        "allocation_id",
        "task_keys",
        "cpus",
        "memory_mib",
        "gpus",
        "time_limit",
        "sbatch_argv",
        "script_digest",
        "argv_digest",
    }
    if set(allocation) != expected:
        raise PlanError("plan allocation keys are invalid")
    allocation_id = allocation["allocation_id"]
    time_limit = allocation["time_limit"]
    script_digest = allocation["script_digest"]
    argv_digest = allocation["argv_digest"]
    if not isinstance(allocation_id, str) or _HEX_24.fullmatch(allocation_id) is None:
        raise PlanError("plan allocation identity is invalid")
    task_keys = _plan_strings(allocation["task_keys"], name="task_keys", nonempty=True)
    if len(set(task_keys)) != len(task_keys):
        raise PlanError("plan allocation task keys are not unique")
    _plan_integer(allocation["cpus"], minimum=1, name="allocation cpus")
    _plan_integer(allocation["memory_mib"], minimum=1, name="allocation memory_mib")
    _plan_integer(allocation["gpus"], minimum=0, name="allocation gpus")
    try:
        effective_seconds = _duration_seconds(time_limit, name="time_limit")
    except ConfigurationError as error:
        raise PlanError("plan allocation time_limit is invalid") from error
    if not isinstance(time_limit, str) or effective_seconds % 60:
        raise PlanError("plan allocation time_limit must be an effective minute duration")
    argv = _plan_strings(allocation["sbatch_argv"], name="sbatch_argv", nonempty=True)
    if not isinstance(script_digest, str) or _HEX_64.fullmatch(script_digest) is None:
        raise PlanError("plan allocation script digest is invalid")
    if (
        not isinstance(argv_digest, str)
        or _HEX_64.fullmatch(argv_digest) is None
        or argv_digest != _digest(argv)
    ):
        raise PlanError("plan allocation command digest is invalid")


def restore_plan(campaign: Campaign, document: object) -> SubmissionPlan:
    mapping = _validate_plan_document(document)
    try:
        completed = cast(list[str], mapping["completed"])
        retry = cast(list[str], mapping["retry"])
        tasks_per = cast(int | None, mapping["tasks_per_allocation"])
        target = _target_from_dict(mapping["target"])
        resources = _resource_from_dict(mapping["resources"])
    except KeyError as error:
        raise PlanError("plan document is incomplete") from error
    plan = campaign.plan(
        target,
        resources,
        completed=completed,
        retry=retry,
        tasks_per_allocation=tasks_per,
    )
    if plan_document(plan) != mapping:
        raise PlanError("plan document is stale, foreign, or changed")
    return plan


def validate_plan(plan: SubmissionPlan) -> tuple[_ValidationResult, ...]:
    seen: set[tuple[int, int, int, int, str]] = set()
    results: list[_ValidationResult] = []
    for item in plan._allocations:
        allocation = item.allocation
        shape_key = (
            len(allocation.task_keys),
            allocation.cpus,
            allocation.memory_mib,
            allocation.gpus,
            allocation.time_limit,
        )
        if shape_key in seen:
            continue
        seen.add(shape_key)
        result = _slurm.validate_allocation(plan._target, item.argv, item.script)
        if result.returncode != 0:
            raise SubmissionError(
                "Slurm rejected a time-specific validation: "
                + result.stderr.decode("utf-8", "replace").strip()
            )
        shape: dict[str, object] = {
            "task_count": len(allocation.task_keys),
            "cpus": allocation.cpus,
            "memory_mib": allocation.memory_mib,
            "gpus": allocation.gpus,
            "time_limit": allocation.time_limit,
        }
        results.append(
            _ValidationResult(
                shape,
                _digest(shape),
                item.script_digest,
                result.stdout.decode("utf-8", "replace").rstrip("\n"),
                result.stderr.decode("utf-8", "replace").rstrip("\n"),
            )
        )
    return tuple(results)


def validation_document(results: tuple[_ValidationResult, ...]) -> dict[str, object]:
    return {
        "time_specific": True,
        "results": [
            {
                "shape": result.shape,
                "shape_digest": result.shape_digest,
                "script_digest": result.script_digest,
                "controller_stdout": result.controller_stdout,
                "controller_stderr": result.controller_stderr,
            }
            for result in results
        ],
    }
