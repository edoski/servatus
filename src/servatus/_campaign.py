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

_SCHEMA_VERSION = 4
_PLAN_SCHEMA_VERSION = 3
_OPEN = "OPEN"
_SEALED = "SEALED"
_UNRESOLVED = "UNRESOLVED"
_ACCEPTED = "ACCEPTED"
_NOT_SUBMITTED = "NOT_SUBMITTED"
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_GRES = re.compile(r"gpu(?::[A-Za-z0-9][A-Za-z0-9._-]*)?\Z")
_DURATION = re.compile(r"(?:(0|[1-9][0-9]*)-)?([0-9]{2}):([0-5][0-9]):([0-5][0-9])\Z")
_HEX_24 = re.compile(r"[0-9a-f]{24}\Z")
_HEX_32 = re.compile(r"[0-9a-f]{32}\Z")
_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_WINDOW_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"
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
            cpus_per_task=cast(int, values["cpus_per_task"]),
            memory_mib_per_task=cast(int, values["memory_mib_per_task"]),
            gpus_per_task=cast(int, values["gpus_per_task"]),
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
        return cls(
            host=cast(str, values["host"]),
            slurm_bin=cast(PurePosixPath, values["slurm_bin"]),
            apptainer=cast(PurePosixPath, values["apptainer"]),
            image=cast(PurePosixPath, values["image"]),
            work_root=cast(PurePosixPath, values["work_root"]),
            log_root=cast(PurePosixPath, values["log_root"]),
            partitions=tuple(cast(list[str], raw_partitions)),
            account=cast(str | None, values.get("account")),
            qos=cast(str | None, values.get("qos")),
            constraint=cast(str | None, values.get("constraint")),
            gpu_gres=cast(str | None, values.get("gpu_gres")),
            max_tasks_per_allocation=cast(int, values["max_tasks_per_allocation"]),
            max_cpus_per_allocation=cast(int, values["max_cpus_per_allocation"]),
            max_memory_mib_per_allocation=cast(int, values["max_memory_mib_per_allocation"]),
            max_gpus_per_allocation=cast(int, values["max_gpus_per_allocation"]),
            max_time_limit=cast(str, values["max_time_limit"]),
            max_allocations_per_submit=cast(int, values["max_allocations_per_submit"]),
            max_script_bytes=cast(int, values["max_script_bytes"]),
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
    unaccepted_task_keys: tuple[str, ...]
    receipts: tuple[JobReceipt, ...]
    ambiguous_allocation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ValidationResult:
    task_count: int
    cpus: int
    memory_mib: int
    gpus: int
    time_limit: str
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


def _reconciliation_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or len(value) != 19:
        raise TaskConflict("campaign attempt reconciliation window is invalid")
    try:
        parsed = datetime.strptime(value, _WINDOW_TIMESTAMP_FORMAT)
    except ValueError as error:
        raise TaskConflict("campaign attempt reconciliation window is invalid") from error
    if parsed.strftime(_WINDOW_TIMESTAMP_FORMAT) != value:
        raise TaskConflict("campaign attempt reconciliation window is invalid")
    return parsed


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


def _tasks_from_state(state: dict[str, object]) -> tuple[Task, ...]:
    return tuple(_task_from_record(record) for record in cast(list[object], state["tasks"]))


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
        return SlurmTarget(
            host=cast(str, mapping["host"]),
            slurm_bin=cast(PurePosixPath, mapping["slurm_bin"]),
            apptainer=cast(PurePosixPath, mapping["apptainer"]),
            image=cast(PurePosixPath, mapping["image"]),
            work_root=cast(PurePosixPath, mapping["work_root"]),
            log_root=cast(PurePosixPath, mapping["log_root"]),
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
    except (TypeError, ConfigurationError) as error:
        raise PlanError("plan target is invalid") from error


def _resource_from_dict(value: object) -> ResourceRequest:
    if not isinstance(value, dict):
        raise PlanError("plan resources are invalid")
    mapping = cast(dict[str, object], value)
    if frozenset(mapping) != ResourceRequest._KEYS:
        raise PlanError("plan resource keys are invalid")
    try:
        return ResourceRequest(
            cpus_per_task=cast(int, mapping["cpus_per_task"]),
            memory_mib_per_task=cast(int, mapping["memory_mib_per_task"]),
            gpus_per_task=cast(int, mapping["gpus_per_task"]),
            time_limit=cast(str, mapping["time_limit"]),
        )
    except ConfigurationError as error:
        raise PlanError("plan resources are invalid") from error


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
    def __init__(self, path: Path, entry: os.stat_result) -> None:
        self._path = path
        self._entry = entry

    @classmethod
    def open(cls, path: Path, tasks: Sequence[Task]) -> Self:
        frozen = tuple(tasks)
        keys = [task.key for task in frozen]
        if len(set(keys)) != len(keys):
            raise ConfigurationError("Task keys must be unique")
        campaign_path, entry = _open_or_create_directory(path)
        campaign = cls(campaign_path, entry)
        with campaign._locked_state(create=True) as state:
            if state is None:
                campaign._write_state(
                    {
                        "schema_version": _SCHEMA_VERSION,
                        "campaign_id": os.urandom(16).hex(),
                        "revision": 0,
                        "phase": _OPEN,
                        "tasks": [_task_record(task) for task in frozen],
                        "lineage": None,
                        "attempts": [],
                    }
                )
            else:
                stored = _tasks_from_state(state)
                if frozen[: len(stored)] != stored:
                    raise TaskConflict("campaign tasks may only gain an exact ordered suffix")
                if len(frozen) > len(stored):
                    if state["phase"] == _SEALED:
                        raise TaskConflict("sealed campaign tasks cannot change")
                    cast(list[object], state["tasks"]).extend(
                        _task_record(task) for task in frozen[len(stored) :]
                    )
                    state["revision"] = cast(int, state["revision"]) + 1
                    campaign._write_state(state)
        return campaign

    @classmethod
    def load(cls, path: Path) -> Self:
        campaign_path, entry = _open_existing_directory(path)
        campaign = cls(campaign_path, entry)
        campaign._read_state()
        return campaign

    @property
    def tasks(self) -> tuple[Task, ...]:
        return _tasks_from_state(self._read_state())

    def seal(self) -> None:
        with self._locked_state() as state:
            assert state is not None
            if state["phase"] == _SEALED:
                return
            state["phase"] = _SEALED
            state["revision"] = cast(int, state["revision"]) + 1
            self._write_state(state)

    def plan(
        self,
        target: SlurmTarget,
        resources: ResourceRequest,
        *,
        completed: Collection[str] = (),
        retry: Collection[str] = (),
        tasks_per_allocation: int | None = None,
    ) -> SubmissionPlan:
        state = self._read_state()
        tasks = _tasks_from_state(state)
        ambiguous = _ambiguous_ids(state)
        if ambiguous:
            raise AmbiguousSubmission("resolve ambiguous allocation intent before planning")
        known = tuple(task.key for task in tasks)
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
            for task in tasks
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
            script = _slurm.render_script(target, resources, group)
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
        self._verify_plan(plan, operation="submit")
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

    def validate(self, plan: SubmissionPlan) -> tuple[ValidationResult, ...]:
        self._verify_plan(plan, operation="validate")
        return _validate_plan(plan)

    def reconcile(self, allocation_id: str) -> JobReceipt:
        state = self._read_state()
        attempt = _unresolved_attempt(state, allocation_id)
        lineage = _validate_lineage(state["lineage"])
        if lineage is None:
            raise ReconciliationError("allocation intent has no target lineage")
        target, _ = lineage
        match = _slurm.query_identity(
            target,
            job_name=f"servatus-{allocation_id}",
            window_start=cast(str, attempt["window_start"]),
            window_end=cast(str, attempt["window_end"]),
        )
        receipt = JobReceipt(
            allocation_id,
            match.job_id,
            match.cluster,
            tuple(cast(list[str], attempt["task_keys"])),
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
            attempt = _unresolved_attempt(state, allocation_id)
            if job_id is None:
                attempt["acceptance"] = {"status": _NOT_SUBMITTED}
            else:
                attempt["acceptance"] = {
                    "status": _ACCEPTED,
                    "job_id": job_id,
                    "cluster": cluster,
                }
            state["revision"] = cast(int, state["revision"]) + 1
            self._write_state(state)

    def status(self) -> CampaignStatus:
        state = self._read_state()
        tasks = _tasks_from_state(state)
        receipts = _receipt_values(state)
        accepted = {key for receipt in receipts for key in receipt.task_keys}
        return CampaignStatus(
            tuple(task.key for task in tasks if task.key not in accepted),
            receipts,
            tuple(_ambiguous_ids(state)),
        )

    def _verify_plan(self, plan: SubmissionPlan, *, operation: str) -> None:
        state = self._read_state()
        if plan._campaign_id != state["campaign_id"]:
            raise PlanError(f"{operation} plan belongs to another campaign")
        if plan._state_revision != state["revision"]:
            raise PlanError(f"{operation} plan is stale")
        for allocation in plan._allocations:
            if allocation.script_digest != hashlib.sha256(allocation.script).hexdigest():
                raise PlanError(f"{operation} plan script was changed")
        if plan.digest != _digest(_plan_semantics(plan)):
            raise PlanError(f"{operation} plan was changed")
        if _ambiguous_ids(state):
            raise AmbiguousSubmission(f"resolve ambiguous allocation intent before {operation}")
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
            lineage = cast(dict[str, object], state["lineage"])
            cast(list[dict[str, object]], state["attempts"]).append(
                {
                    "allocation_id": allocation.allocation_id,
                    "task_keys": list(allocation.allocation.task_keys),
                    "campaign_revision": plan._state_revision,
                    "retry_task_keys": [
                        key for key in allocation.allocation.task_keys if key in plan._retry
                    ],
                    "plan_digest": plan.digest,
                    "script_digest": allocation.script_digest,
                    "target_digest": lineage["target_digest"],
                    "resource_digest": lineage["resource_digest"],
                    "allocation": {
                        "cpus": allocation.allocation.cpus,
                        "memory_mib": allocation.allocation.memory_mib,
                        "gpus": allocation.allocation.gpus,
                        "time_limit": allocation.allocation.time_limit,
                    },
                    "sbatch_argv": list(allocation.argv),
                    "window_start": (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S"),
                    "window_end": (now + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S"),
                    "acceptance": {"status": _UNRESOLVED},
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
            attempt = _unresolved_attempt(state, receipt.allocation_id)
            attempt["acceptance"] = {
                "status": _ACCEPTED,
                "job_id": receipt.job_id,
                "cluster": receipt.cluster,
            }
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
    for attempt in cast(list[dict[str, object]], state["attempts"]):
        acceptance = cast(dict[str, object], attempt["acceptance"])
        if acceptance["status"] != _ACCEPTED:
            continue
        values.append(
            JobReceipt(
                cast(str, attempt["allocation_id"]),
                cast(int, acceptance["job_id"]),
                cast(str | None, acceptance["cluster"]),
                tuple(cast(list[str], attempt["task_keys"])),
            )
        )
    return tuple(values)


def _ambiguous_ids(state: dict[str, object]) -> list[str]:
    return [
        cast(str, attempt["allocation_id"])
        for attempt in cast(list[dict[str, object]], state["attempts"])
        if cast(dict[str, object], attempt["acceptance"])["status"] == _UNRESOLVED
    ]


def _unresolved_attempt(state: dict[str, object], allocation_id: str) -> dict[str, object]:
    if allocation_id not in _ambiguous_ids(state):
        raise ReconciliationError("allocation is not an unresolved ambiguous intent")
    for attempt in cast(list[dict[str, object]], state["attempts"]):
        if attempt["allocation_id"] == allocation_id:
            return attempt
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
        "phase",
        "tasks",
        "lineage",
        "attempts",
    }
    if (
        set(state) != keys
        or type(state.get("schema_version")) is not int
        or state["schema_version"] != _SCHEMA_VERSION
    ):
        raise TaskConflict("campaign state schema is unsupported")
    if (
        not isinstance(state["campaign_id"], str)
        or _HEX_32.fullmatch(state["campaign_id"]) is None
        or not isinstance(state["revision"], int)
        or isinstance(state["revision"], bool)
        or state["revision"] < 0
        or not isinstance(state["phase"], str)
        or state["phase"] not in {_OPEN, _SEALED}
        or not isinstance(state["tasks"], list)
        or not isinstance(state["attempts"], list)
    ):
        raise TaskConflict("campaign state values are invalid")
    if state["phase"] == _SEALED and state["revision"] == 0:
        raise TaskConflict("campaign roster phase is invalid for its revision")
    task_keys = tuple(task.key for task in _tasks_from_state(state))
    if len(set(task_keys)) != len(task_keys):
        raise TaskConflict("campaign task keys are not unique")
    lineage = _validate_lineage(state["lineage"])
    _validate_attempts(
        cast(list[object], state["attempts"]),
        task_keys,
        lineage,
        state["revision"],
        sealed=state["phase"] == _SEALED,
    )


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


def _validate_attempts(
    value: list[object],
    task_keys: tuple[str, ...],
    lineage: tuple[SlurmTarget, ResourceRequest] | None,
    state_revision: int,
    *,
    sealed: bool,
) -> None:
    expected = {
        "allocation_id",
        "task_keys",
        "campaign_revision",
        "retry_task_keys",
        "plan_digest",
        "script_digest",
        "target_digest",
        "resource_digest",
        "allocation",
        "sbatch_argv",
        "window_start",
        "window_end",
        "acceptance",
    }
    allocation_ids: set[str] = set()
    accepted_task_keys: set[str] = set()
    plan_revisions: dict[str, int] = {}
    group_revision: int | None = None
    group_plan_digest: str | None = None
    group_end_revision = 0
    unexplained_revisions = 0
    if not value:
        if lineage is not None:
            raise TaskConflict("campaign lineage has no attempt history")
        return
    if lineage is None:
        raise TaskConflict("campaign attempt has no resource lineage")
    target, resources = lineage
    lineage_values = _lineage(target, resources)
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise TaskConflict("campaign attempt is invalid")
        attempt = cast(dict[str, object], raw)
        allocation_id = attempt.get("allocation_id")
        raw_keys = attempt.get("task_keys")
        campaign_revision = attempt.get("campaign_revision")
        if (
            set(attempt) != expected
            or not isinstance(allocation_id, str)
            or _HEX_24.fullmatch(allocation_id) is None
            or allocation_id in allocation_ids
            or not isinstance(raw_keys, list)
            or type(campaign_revision) is not int
            or campaign_revision < 0
            or campaign_revision >= state_revision
        ):
            raise TaskConflict("campaign attempt is invalid")
        attempt_keys = tuple(cast(list[object], raw_keys))
        if (
            not attempt_keys
            or any(not isinstance(key, str) or key not in task_keys for key in attempt_keys)
            or len(set(attempt_keys)) != len(attempt_keys)
        ):
            raise TaskConflict("campaign attempt task keys are invalid")
        typed_attempt_keys = cast(tuple[str, ...], attempt_keys)
        if tuple(key for key in task_keys if key in set(typed_attempt_keys)) != typed_attempt_keys:
            raise TaskConflict("campaign attempt task keys are out of order")
        for name in ("plan_digest", "script_digest"):
            digest = attempt[name]
            if not isinstance(digest, str) or _HEX_64.fullmatch(digest) is None:
                raise TaskConflict("campaign attempt digest is invalid")
        plan_digest = cast(str, attempt["plan_digest"])
        known_revision = plan_revisions.setdefault(plan_digest, campaign_revision)
        if known_revision != campaign_revision:
            raise TaskConflict("campaign attempt plan revision is inconsistent")
        if group_revision is None:
            unexplained_revisions = campaign_revision
            group_revision = campaign_revision
            group_plan_digest = plan_digest
            group_end_revision = campaign_revision
        elif campaign_revision == group_revision:
            if plan_digest != group_plan_digest:
                raise TaskConflict("campaign attempt plan group is invalid")
        else:
            if campaign_revision < group_end_revision:
                raise TaskConflict("campaign attempt revision precedes durable history")
            unexplained_revisions += campaign_revision - group_end_revision
            group_revision = campaign_revision
            group_plan_digest = plan_digest
            group_end_revision = campaign_revision
        if (
            attempt["target_digest"] != lineage_values["target_digest"]
            or attempt["resource_digest"] != lineage_values["resource_digest"]
        ):
            raise TaskConflict("campaign attempt lineage is invalid")
        retry_keys = attempt["retry_task_keys"]
        if not isinstance(retry_keys, list):
            raise TaskConflict("campaign attempt retry keys are invalid")
        typed_retry_keys = tuple(cast(list[object], retry_keys))
        if any(not isinstance(key, str) for key in typed_retry_keys):
            raise TaskConflict("campaign attempt retry keys are invalid")
        typed_retries = cast(tuple[str, ...], typed_retry_keys)
        expected_retries = tuple(key for key in typed_attempt_keys if key in accepted_task_keys)
        if typed_retries != expected_retries:
            raise TaskConflict("campaign attempt retry keys do not match accepted history")
        allocation = attempt["allocation"]
        argv = attempt["sbatch_argv"]
        if not isinstance(allocation, dict) or not isinstance(argv, list):
            raise TaskConflict("campaign attempt provenance is invalid")
        totals = cast(dict[str, object], allocation)
        typed_argv = cast(list[object], argv)
        task_count = len(attempt_keys)
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
        numeric_totals = {"cpus": 1, "memory_mib": 1, "gpus": 0}
        if (
            any(
                type(totals.get(name)) is not int or cast(int, totals[name]) < minimum
                for name, minimum in numeric_totals.items()
            )
            or totals != expected_totals
            or typed_argv != expected_argv
        ):
            raise TaskConflict("campaign attempt provenance is invalid")
        window_start = _reconciliation_timestamp(attempt["window_start"])
        window_end = _reconciliation_timestamp(attempt["window_end"])
        if window_start >= window_end:
            raise TaskConflict("campaign attempt reconciliation window is not ordered")
        acceptance = attempt["acceptance"]
        if not isinstance(acceptance, dict):
            raise TaskConflict("campaign attempt acceptance is invalid")
        typed_acceptance = cast(dict[str, object], acceptance)
        status = typed_acceptance.get("status")
        if isinstance(status, str) and status in {_UNRESOLVED, _NOT_SUBMITTED}:
            if set(typed_acceptance) != {"status"}:
                raise TaskConflict("campaign attempt acceptance is invalid")
            if status == _UNRESOLVED and index != len(value) - 1:
                raise TaskConflict("campaign unresolved attempt must be final")
            if status == _NOT_SUBMITTED:
                group_end_revision += 1
        elif status == _ACCEPTED:
            job_id = typed_acceptance.get("job_id")
            cluster = typed_acceptance.get("cluster")
            if (
                set(typed_acceptance) != {"status", "job_id", "cluster"}
                or type(job_id) is not int
                or job_id <= 0
                or (
                    cluster is not None
                    and (not isinstance(cluster, str) or _TOKEN.fullmatch(cluster) is None)
                )
            ):
                raise TaskConflict("campaign attempt acceptance is invalid")
            accepted_task_keys.update(typed_attempt_keys)
            group_end_revision += 1
        else:
            raise TaskConflict("campaign attempt acceptance is invalid")
        allocation_ids.add(allocation_id)
        group_end_revision += 1
    if state_revision < group_end_revision:
        raise TaskConflict("campaign revision is behind durable attempt history")
    unexplained_revisions += state_revision - group_end_revision
    if sealed and unexplained_revisions < 1:
        raise TaskConflict("campaign roster phase is invalid for its revision history")


def _plan_strings(value: object, *, name: str) -> list[str]:
    if not isinstance(value, list):
        raise PlanError(f"plan {name} must be an array of strings")
    strings: list[str] = []
    for item in cast(list[object], value):
        if not isinstance(item, str) or "\0" in item:
            raise PlanError(f"plan {name} must be an array of strings")
        strings.append(item)
    return strings


def _plan_inputs(
    document: object,
) -> tuple[
    dict[str, object],
    SlurmTarget,
    ResourceRequest,
    list[str],
    list[str],
    int | None,
]:
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
    target = _target_from_dict(mapping["target"])
    resources = _resource_from_dict(mapping["resources"])
    completed = _plan_strings(mapping["completed"], name="completed")
    retry = _plan_strings(mapping["retry"], name="retry")
    if len(set(completed)) != len(completed) or len(set(retry)) != len(retry):
        raise PlanError("plan selections contain duplicate task keys")
    tasks_per = mapping["tasks_per_allocation"]
    if tasks_per is not None and (
        isinstance(tasks_per, bool) or not isinstance(tasks_per, int) or tasks_per < 1
    ):
        raise PlanError("plan tasks_per_allocation must be an integer >= 1")
    return mapping, target, resources, completed, retry, tasks_per


def restore_plan(campaign: Campaign, document: object) -> SubmissionPlan:
    mapping, target, resources, completed, retry, tasks_per = _plan_inputs(document)
    plan = campaign.plan(
        target,
        resources,
        completed=completed,
        retry=retry,
        tasks_per_allocation=tasks_per,
    )
    try:
        changed = _canonical(plan_document(plan)) != _canonical(mapping)
    except (TypeError, ValueError) as error:
        raise PlanError("plan document contains invalid JSON values") from error
    if changed:
        raise PlanError("plan document is stale, foreign, or changed")
    return plan


def _validate_plan(plan: SubmissionPlan) -> tuple[ValidationResult, ...]:
    seen: set[tuple[int, int, int, int, str]] = set()
    results: list[ValidationResult] = []
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
        result = _slurm._run_ssh(plan._target, (*item.argv, "--test-only"), item.script)
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
            ValidationResult(
                len(allocation.task_keys),
                allocation.cpus,
                allocation.memory_mib,
                allocation.gpus,
                allocation.time_limit,
                _digest(shape),
                item.script_digest,
                result.stdout.decode("utf-8", "replace").rstrip("\n"),
                result.stderr.decode("utf-8", "replace").rstrip("\n"),
            )
        )
    return tuple(results)


def validation_document(results: tuple[ValidationResult, ...]) -> dict[str, object]:
    return {
        "time_specific": True,
        "results": [
            {
                "shape": {
                    "task_count": result.task_count,
                    "cpus": result.cpus,
                    "memory_mib": result.memory_mib,
                    "gpus": result.gpus,
                    "time_limit": result.time_limit,
                },
                "shape_digest": result.shape_digest,
                "script_digest": result.script_digest,
                "controller_stdout": result.controller_stdout,
                "controller_stderr": result.controller_stderr,
            }
            for result in results
        ],
    }
