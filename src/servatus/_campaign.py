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
from collections.abc import Callable, Collection, Generator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import ClassVar, Self, cast

from . import _slurm
from ._errors import (
    AmbiguousSubmission,
    ConfigurationError,
    ObservationError,
    PlanError,
    ReconciliationError,
    SubmissionError,
    TaskConflict,
)

_SCHEMA_VERSION = 4
_PLAN_SCHEMA_VERSION = 4
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
_MAX_LOG_BYTES = 1024 * 1024


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


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{name} must be a nonempty string")
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


def _resource_from_values(mapping: dict[str, object]) -> ResourceRequest:
    return ResourceRequest(
        cpus_per_task=cast(int, mapping["cpus_per_task"]),
        memory_mib_per_task=cast(int, mapping["memory_mib_per_task"]),
        gpus_per_task=cast(int, mapping["gpus_per_task"]),
        time_limit=cast(str, mapping["time_limit"]),
    )


def _profile_resources(values: object) -> ResourceRequest:
    if not isinstance(values, dict):
        raise ConfigurationError("resources must be a table")
    mapping = cast(dict[str, object], values)
    unknown = mapping.keys() - ResourceRequest._KEYS
    missing = ResourceRequest._KEYS - mapping.keys()
    if unknown:
        raise ConfigurationError(f"unknown resource keys: {', '.join(sorted(unknown))}")
    if missing:
        raise ConfigurationError(f"missing resource keys: {', '.join(sorted(missing))}")
    return _resource_from_values(mapping)


def _target_from_values(mapping: dict[str, object]) -> SlurmTarget:
    raw_partitions = mapping["partitions"]
    if not isinstance(raw_partitions, list):
        raise ConfigurationError("partitions must be an array of strings")
    return SlurmTarget(
        host=cast(str, mapping["host"]),
        slurm_bin=cast(PurePosixPath, mapping["slurm_bin"]),
        apptainer=cast(PurePosixPath, mapping["apptainer"]),
        image=cast(PurePosixPath, mapping["image"]),
        work_root=cast(PurePosixPath, mapping["work_root"]),
        log_root=cast(PurePosixPath, mapping["log_root"]),
        partitions=tuple(cast(list[str], raw_partitions)),
        account=cast(str | None, mapping.get("account")),
        qos=cast(str | None, mapping.get("qos")),
        constraint=cast(str | None, mapping.get("constraint")),
        gpu_gres=cast(str | None, mapping.get("gpu_gres")),
        max_tasks_per_allocation=cast(int, mapping["max_tasks_per_allocation"]),
        max_cpus_per_allocation=cast(int, mapping["max_cpus_per_allocation"]),
        max_memory_mib_per_allocation=cast(int, mapping["max_memory_mib_per_allocation"]),
        max_gpus_per_allocation=cast(int, mapping["max_gpus_per_allocation"]),
        max_time_limit=cast(str, mapping["max_time_limit"]),
        max_allocations_per_submit=cast(int, mapping["max_allocations_per_submit"]),
        max_script_bytes=cast(int, mapping["max_script_bytes"]),
    )


def _profile_target(values: object) -> SlurmTarget:
    if not isinstance(values, dict):
        raise ConfigurationError("target must be a table")
    mapping = cast(dict[str, object], values)
    allowed = SlurmTarget._REQUIRED | SlurmTarget._OPTIONAL
    unknown = mapping.keys() - allowed
    missing = SlurmTarget._REQUIRED - mapping.keys()
    if unknown:
        raise ConfigurationError(f"unknown target keys: {', '.join(sorted(unknown))}")
    if missing:
        raise ConfigurationError(f"missing target keys: {', '.join(sorted(missing))}")
    return _target_from_values(mapping)


@dataclass(frozen=True, slots=True)
class Profile:
    label: str
    target: SlurmTarget
    resources: ResourceRequest

    def __post_init__(self) -> None:
        _nonempty_string(self.label, name="profile label")
        _configuration(isinstance(self.target, SlurmTarget), "profile target is invalid")
        _configuration(isinstance(self.resources, ResourceRequest), "profile resources are invalid")

    @classmethod
    def load(cls, path: Path, name: str | None = None) -> Self:
        document = _read_toml(path)
        if document.keys() - {"profiles", "default_profile"}:
            unknown = document.keys() - {"profiles", "default_profile"}
            raise ConfigurationError(f"unknown profile document keys: {', '.join(sorted(unknown))}")
        profiles = document.get("profiles")
        if not isinstance(profiles, dict) or not profiles:
            raise ConfigurationError("profiles must be a nonempty table")

        loaded: dict[str, Self] = {}
        for label, raw in cast(dict[str, object], profiles).items():
            _nonempty_string(label, name="profile label")
            if not isinstance(raw, dict):
                raise ConfigurationError(f"profile {label!r} must contain target and resources")
            mapping = cast(dict[str, object], raw)
            if set(mapping) != {"target", "resources"}:
                raise ConfigurationError(f"profile {label!r} must contain target and resources")
            loaded[label] = cls(
                label,
                _profile_target(mapping["target"]),
                _profile_resources(mapping["resources"]),
            )

        default = document.get("default_profile")
        if default is not None:
            _nonempty_string(default, name="default_profile")
            if cast(str, default) not in loaded:
                raise ConfigurationError("default_profile does not name a declared profile")
        selected = name if name is not None else cast(str | None, default)
        if selected is None:
            raise ConfigurationError("profile selection is required")
        _nonempty_string(selected, name="profile name")
        try:
            return loaded[selected]
        except KeyError:
            raise ConfigurationError(f"profile {selected!r} is not declared") from None


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
    _profile: Profile
    _roster_digest: str
    _view: CampaignView
    _view_digest: str
    _selected: tuple[str, ...]
    _excluded: tuple[str, ...]
    _retry: tuple[str, ...]
    _duplicate_risk: tuple[str, ...]
    _tasks_per_allocation: int | None
    _allocations: tuple[_AllocationPlan, ...]
    _digest: str

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def allocations(self) -> tuple[PlannedAllocation, ...]:
        return tuple(item.allocation for item in self._allocations)

    @property
    def profile(self) -> Profile:
        return self._profile

    @property
    def selected_task_keys(self) -> tuple[str, ...]:
        return self._selected

    @property
    def excluded_task_keys(self) -> tuple[str, ...]:
        return self._excluded

    @property
    def retry_task_keys(self) -> tuple[str, ...]:
        return self._retry

    @property
    def duplicate_risk_task_keys(self) -> tuple[str, ...]:
        return self._duplicate_risk

    @property
    def warnings(self) -> tuple[str, ...]:
        if not self._duplicate_risk:
            return ()
        return (
            "duplicate execution risk accepted for unknown prior work: "
            + ", ".join(self._duplicate_risk),
        )


@dataclass(frozen=True, slots=True)
class JobReceipt:
    allocation_id: str
    job_id: int
    cluster: str | None
    task_keys: tuple[str, ...]

    def __str__(self) -> str:
        suffix = "" if self.cluster is None else f";{self.cluster}"
        return f"{self.job_id}{suffix}"


class ResultState(StrEnum):
    UNOBSERVED = "UNOBSERVED"
    MISSING = "MISSING"
    VALID = "VALID"


class AcceptanceState(StrEnum):
    UNRESOLVED = _UNRESOLVED
    ACCEPTED = _ACCEPTED
    NOT_SUBMITTED = _NOT_SUBMITTED


AllocationState = _slurm.AllocationState


ResultProbe = Callable[[Task], bool]


@dataclass(frozen=True, slots=True)
class AllocationEvidence:
    state: AllocationState
    raw_state: str | None
    accounting_state: str | None
    exit_code: str | None
    reason: str | None
    started_at: str | None
    ended_at: str | None
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class AttemptEvidence:
    allocation_id: str
    task_keys: tuple[str, ...]
    retry_task_keys: tuple[str, ...]
    duplicate_risk_task_keys: tuple[str, ...]
    profile_label: str
    acceptance: AcceptanceState
    receipt: JobReceipt | None
    allocation: AllocationEvidence | None


@dataclass(frozen=True, slots=True)
class TaskEvidence:
    key: str
    result: ResultState
    result_observed_at: datetime | None
    current_attempt_id: str | None
    execution: AllocationState | None
    acceptance_ambiguous: bool


@dataclass(frozen=True, slots=True)
class CampaignView:
    campaign_id: str
    revision: int
    sealed: bool
    tasks: tuple[TaskEvidence, ...]
    attempts: tuple[AttemptEvidence, ...]
    scheduler_observed: bool
    observed_at: datetime
    results_ready: bool
    quiescent: bool


@dataclass(frozen=True, slots=True)
class LogSnapshot:
    content: bytes = field(repr=False)
    truncated: bool
    observed_at: datetime


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


def _validate_revision_history(
    state_revision: int,
    mutation_revisions: set[int],
    plan_requirements: list[tuple[int, int]],
    task_count: int,
    *,
    sealed: bool,
) -> None:
    missing_count = state_revision - len(mutation_revisions)
    if missing_count < int(sealed) or missing_count > task_count + int(sealed):
        raise TaskConflict("campaign revision cannot be explained by roster history")
    append_revisions: list[int] = []
    candidate = 1
    for mutation_revision in sorted(mutation_revisions):
        while candidate < mutation_revision:
            append_revisions.append(candidate)
            candidate += 1
        candidate = mutation_revision + 1
    while candidate <= state_revision:
        append_revisions.append(candidate)
        candidate += 1
    if sealed:
        append_revisions.pop()
    append_index = 0
    for plan_revision, required_prefix in plan_requirements:
        while (
            append_index < len(append_revisions) and append_revisions[append_index] <= plan_revision
        ):
            append_index += 1
        future_appends = len(append_revisions) - append_index
        if required_prefix > task_count - future_appends:
            raise TaskConflict("campaign revision cannot be explained by roster history")


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
        return _target_from_values(mapping)
    except ConfigurationError as error:
        raise PlanError("plan target is invalid") from error


def _resource_from_dict(value: object) -> ResourceRequest:
    if not isinstance(value, dict):
        raise PlanError("plan resources are invalid")
    mapping = cast(dict[str, object], value)
    if frozenset(mapping) != ResourceRequest._KEYS:
        raise PlanError("plan resource keys are invalid")
    try:
        return _resource_from_values(mapping)
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


def _timestamp_text(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _allocation_evidence_dict(value: AllocationEvidence | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "state": value.state.value,
        "raw_state": value.raw_state,
        "accounting_state": value.accounting_state,
        "exit_code": value.exit_code,
        "reason": value.reason,
        "started_at": value.started_at,
        "ended_at": value.ended_at,
        "observed_at": _timestamp_text(value.observed_at),
    }


def _view_semantics(view: CampaignView) -> dict[str, object]:
    return {
        "campaign_id": view.campaign_id,
        "revision": view.revision,
        "sealed": view.sealed,
        "tasks": [
            {
                "key": task.key,
                "result": task.result.value,
                "result_observed_at": _timestamp_text(task.result_observed_at),
                "current_attempt_id": task.current_attempt_id,
                "execution": None if task.execution is None else task.execution.value,
                "acceptance_ambiguous": task.acceptance_ambiguous,
            }
            for task in view.tasks
        ],
        "attempts": [
            {
                "allocation_id": attempt.allocation_id,
                "task_keys": list(attempt.task_keys),
                "retry_task_keys": list(attempt.retry_task_keys),
                "duplicate_risk_task_keys": list(attempt.duplicate_risk_task_keys),
                "profile_label": attempt.profile_label,
                "acceptance": attempt.acceptance.value,
                "receipt": (
                    None
                    if attempt.receipt is None
                    else {
                        "allocation_id": attempt.receipt.allocation_id,
                        "job_id": attempt.receipt.job_id,
                        "cluster": attempt.receipt.cluster,
                        "task_keys": list(attempt.receipt.task_keys),
                    }
                ),
                "allocation": _allocation_evidence_dict(attempt.allocation),
            }
            for attempt in view.attempts
        ],
        "scheduler_observed": view.scheduler_observed,
        "observed_at": _timestamp_text(view.observed_at),
        "results_ready": view.results_ready,
        "quiescent": view.quiescent,
    }


def campaign_view_document(view: CampaignView) -> dict[str, object]:
    return _view_semantics(view)


def _plan_semantics(plan: SubmissionPlan) -> dict[str, object]:
    return {
        "campaign_id": plan._campaign_id,
        "state_revision": plan._state_revision,
        "profile_label": plan._profile.label,
        "target": _target_dict(plan._profile.target),
        "resources": _resource_dict(plan._profile.resources),
        "roster_digest": plan._roster_digest,
        "view": _view_semantics(plan._view),
        "view_digest": plan._view_digest,
        "selected": list(plan._selected),
        "excluded": list(plan._excluded),
        "retry": list(plan._retry),
        "duplicate_risk": list(plan._duplicate_risk),
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

    def inspect(
        self,
        probe: ResultProbe | None = None,
        *,
        scheduler: bool = True,
    ) -> CampaignView:
        state = self._read_state()
        campaign_id = cast(str, state["campaign_id"])
        revision = cast(int, state["revision"])
        tasks = _tasks_from_state(state)

        result_values: list[tuple[ResultState, datetime | None]] = []
        for task in tasks:
            if probe is None:
                result_values.append((ResultState.UNOBSERVED, None))
                continue
            valid = probe(task)
            if type(valid) is not bool:
                raise TypeError("result probe must return bool")
            result_values.append(
                (ResultState.VALID if valid else ResultState.MISSING, datetime.now(UTC))
            )

        receipt_values = _receipt_values(state)
        receipts = {receipt.allocation_id: receipt for receipt in receipt_values}
        attempt_queries = _attempt_queries(state)
        scheduler_values: tuple[_slurm.SchedulerObservation, ...] = ()
        if scheduler and attempt_queries:
            lineage = _validate_lineage(state["lineage"])
            assert lineage is not None
            scheduler_values = _slurm.query_attempts(lineage[0], attempt_queries)
        scheduler_observed_at = datetime.now(UTC)
        scheduler_by_allocation = (
            {
                query.allocation_id: observation
                for query, observation in zip(attempt_queries, scheduler_values, strict=True)
            }
            if scheduler
            else {}
        )
        attempts: list[AttemptEvidence] = []
        current: dict[str, AttemptEvidence] = {}
        ambiguous: set[str] = set()
        for attempt in cast(list[dict[str, object]], state["attempts"]):
            acceptance = cast(dict[str, object], attempt["acceptance"])
            status = AcceptanceState(cast(str, acceptance["status"]))
            receipt = receipts.get(cast(str, attempt["allocation_id"]))
            observation = scheduler_by_allocation.get(cast(str, attempt["allocation_id"]))
            allocation = (
                None
                if receipt is None or observation is None
                else AllocationEvidence(
                    observation.state,
                    observation.raw_state,
                    observation.accounting_state,
                    observation.exit_code,
                    observation.reason,
                    observation.started_at,
                    observation.ended_at,
                    scheduler_observed_at,
                )
            )
            evidence = AttemptEvidence(
                cast(str, attempt["allocation_id"]),
                tuple(cast(list[str], attempt["task_keys"])),
                tuple(cast(list[str], attempt["retry_task_keys"])),
                tuple(cast(list[str], attempt["duplicate_risk_task_keys"])),
                cast(str, attempt["profile_label"]),
                status,
                receipt,
                allocation,
            )
            attempts.append(evidence)
            if status is AcceptanceState.ACCEPTED:
                for key in evidence.task_keys:
                    current[key] = evidence
            elif status is AcceptanceState.UNRESOLVED:
                ambiguous.update(evidence.task_keys)
                for key in evidence.task_keys:
                    current[key] = evidence

        after = self._read_state()
        if after["campaign_id"] != campaign_id or after["revision"] != revision:
            raise ObservationError("campaign changed during inspection")

        observed_at = datetime.now(UTC)
        task_items: list[TaskEvidence] = []
        for index, task in enumerate(tasks):
            current_attempt = current.get(task.key)
            task_items.append(
                TaskEvidence(
                    task.key,
                    result_values[index][0],
                    result_values[index][1],
                    None if current_attempt is None else current_attempt.allocation_id,
                    (
                        None
                        if current_attempt is None or current_attempt.allocation is None
                        else current_attempt.allocation.state
                    ),
                    task.key in ambiguous,
                )
            )
        task_evidence = tuple(task_items)
        results_ready = state["phase"] == _SEALED and all(
            item.result is ResultState.VALID for item in task_evidence
        )
        terminal = {
            AllocationState.SUCCEEDED,
            AllocationState.FAILED,
            AllocationState.CANCELLED,
        }
        quiescent = (
            scheduler
            and not ambiguous
            and all(
                attempt.allocation is not None and attempt.allocation.state in terminal
                for attempt in attempts
                if attempt.acceptance is AcceptanceState.ACCEPTED
            )
        )
        return CampaignView(
            campaign_id,
            revision,
            state["phase"] == _SEALED,
            task_evidence,
            tuple(attempts),
            scheduler,
            observed_at,
            results_ready,
            quiescent,
        )

    def read_log(
        self,
        allocation_id: str,
        *,
        task_key: str | None = None,
        max_bytes: int = 65_536,
    ) -> LogSnapshot:
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or not 1 <= max_bytes <= _MAX_LOG_BYTES
        ):
            raise ConfigurationError("max_bytes must be an integer between 1 and 1048576")
        state = self._read_state()
        lineage = _validate_lineage(state["lineage"])
        location: tuple[SlurmTarget, PurePosixPath] | None = None
        if lineage is not None:
            target, _ = lineage
            for attempt in cast(list[dict[str, object]], state["attempts"]):
                acceptance = cast(dict[str, object], attempt["acceptance"])
                if attempt["allocation_id"] != allocation_id or acceptance["status"] != _ACCEPTED:
                    continue
                task_keys = tuple(cast(list[str], attempt["task_keys"]))
                if task_key is not None and task_key not in task_keys:
                    break
                slot = None if task_key is None else task_keys.index(task_key)
                location = (
                    target,
                    _slurm._log_path(
                        target.log_root,
                        cast(str, attempt["allocation_id"]),
                        cast(int, acceptance["job_id"]),
                        slot,
                    ),
                )
                break
        if location is None:
            raise ObservationError("campaign log is unavailable")
        content, truncated = _slurm.read_log_suffix(*location, max_bytes)
        return LogSnapshot(content, truncated, datetime.now(UTC))

    def plan(
        self,
        profile: Profile,
        *,
        view: CampaignView,
        retry: Collection[str] = (),
        allow_duplicate_risk: Collection[str] = (),
        tasks_per_allocation: int | None = None,
    ) -> SubmissionPlan:
        state = self._read_state()
        tasks = _tasks_from_state(state)
        known = tuple(task.key for task in tasks)
        retry_values = _ordered_selection(retry, known, "retry")
        duplicate_risk = _ordered_selection(allow_duplicate_risk, known, "allow_duplicate_risk")
        if any(key not in retry_values for key in duplicate_risk):
            raise PlanError("duplicate-risk acknowledgement requires explicit retry")
        _verify_view(state, view)
        lineage = state["lineage"]
        if lineage is not None and lineage != _lineage(profile.target, profile.resources):
            raise PlanError("campaign is bound to different target or resource semantics")
        selected_keys, excluded_keys = _select_tasks(view, retry_values, duplicate_risk)
        capacity = _capacity(profile.target, profile.resources)
        if tasks_per_allocation is not None:
            requested = _integer(tasks_per_allocation, minimum=1, name="tasks_per_allocation")
            if requested > capacity:
                raise PlanError("tasks_per_allocation exceeds feasible capacity")
            capacity = requested
        selected_key_set = set(selected_keys)
        selected = tuple(task for task in tasks if task.key in selected_key_set)
        groups = _balanced_groups(selected, capacity)
        campaign_id = cast(str, state["campaign_id"])
        revision = cast(int, state["revision"])
        roster_digest = _digest([_task_record(task) for task in tasks])
        view_digest = _digest(_view_semantics(view))
        seed = _digest(
            {
                "campaign_id": campaign_id,
                "revision": revision,
                "profile_label": profile.label,
                "target": _target_dict(profile.target),
                "resources": _resource_dict(profile.resources),
                "roster_digest": roster_digest,
                "view_digest": view_digest,
                "selected": list(selected_keys),
                "excluded": list(excluded_keys),
                "retry": list(retry_values),
                "duplicate_risk": list(duplicate_risk),
                "tasks_per_allocation": tasks_per_allocation,
                "groups": [[task.key for task in group] for group in groups],
            }
        )
        allocations: list[_AllocationPlan] = []
        effective_time_limit = _effective_time_limit(profile.resources.time_limit)
        for index, group in enumerate(groups):
            allocation_id = hashlib.sha256(f"{seed}:{index}".encode()).hexdigest()[:24]
            script = _slurm.render_script(profile.target, profile.resources, group, allocation_id)
            if len(script) > profile.target.max_script_bytes:
                raise PlanError(
                    f"rendered script is {len(script)} bytes; target permits "
                    f"{profile.target.max_script_bytes}"
                )
            argv = _slurm.sbatch_argv(
                profile.target,
                profile.resources,
                len(group),
                allocation_id,
                effective_time_limit,
            )
            public = PlannedAllocation(
                task_keys=tuple(task.key for task in group),
                cpus=len(group) * profile.resources.cpus_per_task,
                memory_mib=len(group) * profile.resources.memory_mib_per_task,
                gpus=len(group) * profile.resources.gpus_per_task,
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
            profile,
            roster_digest,
            view,
            view_digest,
            selected_keys,
            excluded_keys,
            retry_values,
            duplicate_risk,
            tasks_per_allocation,
            tuple(allocations),
            "",
        )
        return SubmissionPlan(
            campaign_id,
            revision,
            profile,
            roster_digest,
            view,
            view_digest,
            selected_keys,
            excluded_keys,
            retry_values,
            duplicate_risk,
            tasks_per_allocation,
            tuple(allocations),
            _digest(_plan_semantics(partial)),
        )

    def submit(
        self, plan: SubmissionPlan, *, probe: ResultProbe | None = None
    ) -> tuple[JobReceipt, ...]:
        self._verify_plan(plan, operation="submit")
        result_aware = any(task.result is not ResultState.UNOBSERVED for task in plan._view.tasks)
        if result_aware and plan._allocations and probe is None:
            raise PlanError("result-aware submission requires the planning result probe")
        expected_revision = plan._state_revision
        receipts: list[JobReceipt] = []
        for allocation in plan._allocations[: plan._profile.target.max_allocations_per_submit]:
            self._refresh_allocation(
                plan,
                allocation,
                expected_revision=expected_revision,
                probe=probe if result_aware else None,
            )
            expected_revision = self._record_intent(
                plan, allocation, expected_revision=expected_revision
            )
            try:
                result = _slurm._run_ssh(plan._profile.target, allocation.argv, allocation.script)
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
                attempt["acceptance"] = {
                    "status": _NOT_SUBMITTED,
                    "outcome_revision": cast(int, state["revision"]) + 1,
                }
            else:
                attempt["acceptance"] = {
                    "status": _ACCEPTED,
                    "job_id": job_id,
                    "cluster": cluster,
                    "outcome_revision": cast(int, state["revision"]) + 1,
                }
            state["revision"] = cast(int, state["revision"]) + 1
            self._write_state(state)

    def _verify_plan(self, plan: SubmissionPlan, *, operation: str) -> None:
        state = self._read_state()
        if plan._campaign_id != state["campaign_id"]:
            raise PlanError(f"{operation} plan belongs to another campaign")
        if plan._state_revision != state["revision"]:
            raise PlanError(f"{operation} plan is stale")
        if plan._roster_digest != _digest(
            [_task_record(task) for task in _tasks_from_state(state)]
        ):
            raise PlanError(f"{operation} plan roster was changed")
        if plan._view_digest != _digest(_view_semantics(plan._view)):
            raise PlanError(f"{operation} plan view was changed")
        _verify_view(state, plan._view)
        selected, excluded = _select_tasks(plan._view, plan._retry, plan._duplicate_risk)
        if selected != plan._selected or excluded != plan._excluded:
            raise PlanError(f"{operation} plan selection was changed")
        for allocation in plan._allocations:
            if allocation.script_digest != hashlib.sha256(allocation.script).hexdigest():
                raise PlanError(f"{operation} plan script was changed")
        if plan.digest != _digest(_plan_semantics(plan)):
            raise PlanError(f"{operation} plan was changed")
        lineage = state["lineage"]
        expected = _lineage(plan._profile.target, plan._profile.resources)
        if lineage is not None and lineage != expected:
            raise PlanError("campaign is bound to different target or resource semantics")

    def _refresh_allocation(
        self,
        plan: SubmissionPlan,
        allocation: _AllocationPlan,
        *,
        expected_revision: int,
        probe: ResultProbe | None,
    ) -> None:
        state = self._read_state()
        if state["campaign_id"] != plan._campaign_id or state["revision"] != expected_revision:
            raise PlanError("campaign changed before submission freshness check")
        current_roster_digest = _digest([_task_record(task) for task in _tasks_from_state(state)])
        if current_roster_digest != plan._roster_digest:
            raise PlanError("campaign roster changed before submission")

        selected = set(allocation.allocation.task_keys)
        if probe is not None:
            tasks = {task.key: task for task in _tasks_from_state(state)}
            for key in allocation.allocation.task_keys:
                valid = probe(tasks[key])
                if type(valid) is not bool:
                    raise TypeError("result probe must return bool")
                if valid:
                    raise PlanError(f"task {key!r} became ineligible before submission")

        attempts = cast(list[dict[str, object]], state["attempts"])
        relevant_queries = tuple(
            (attempt, query)
            for attempt, query in zip(
                (
                    attempt
                    for attempt in attempts
                    if cast(dict[str, object], attempt["acceptance"])["status"] == _ACCEPTED
                ),
                _attempt_queries(state),
                strict=True,
            )
            if selected & set(cast(list[str], attempt["task_keys"]))
        )
        observations = (
            _slurm.query_attempts(
                plan._profile.target, tuple(query for _attempt, query in relevant_queries)
            )
            if relevant_queries
            else ()
        )

        after = self._read_state()
        if after["campaign_id"] != plan._campaign_id or after["revision"] != expected_revision:
            raise PlanError("campaign changed during submission freshness check")
        if any(
            selected & set(cast(list[str], attempt["task_keys"]))
            for attempt in cast(list[dict[str, object]], after["attempts"])
            if cast(dict[str, object], attempt["acceptance"])["status"] == _UNRESOLVED
        ):
            raise PlanError("selected task became acceptance-ambiguous before submission")

        states_by_key: dict[str, list[AllocationState]] = {key: [] for key in selected}
        for (attempt, _query), observation in zip(relevant_queries, observations, strict=True):
            for key in selected & set(cast(list[str], attempt["task_keys"])):
                states_by_key[key].append(observation.state)
        active = {AllocationState.QUEUED, AllocationState.RUNNING}
        retries = set(plan._retry)
        acknowledged = set(plan._duplicate_risk)
        for key in allocation.allocation.task_keys:
            states = states_by_key[key]
            if any(state in active for state in states):
                raise PlanError(f"task {key!r} became active before submission")
            if states and key not in retries:
                raise PlanError(f"task {key!r} became accepted before submission")
            if not states and key in retries:
                raise PlanError(f"task {key!r} lost accepted-attempt evidence")
            if AllocationState.UNKNOWN in states and key not in acknowledged:
                raise PlanError(f"task {key!r} now requires duplicate-risk acknowledgement")
            if AllocationState.UNKNOWN not in states and key in acknowledged:
                raise PlanError(
                    f"task {key!r} no longer matches its duplicate-risk acknowledgement"
                )

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
            allocation_keys = set(allocation.allocation.task_keys)
            conflicting_ambiguity = any(
                allocation_keys & set(cast(list[str], attempt["task_keys"]))
                for attempt in cast(list[dict[str, object]], state["attempts"])
                if cast(dict[str, object], attempt["acceptance"])["status"] == _UNRESOLVED
            )
            if state["revision"] != expected_revision or conflicting_ambiguity:
                raise PlanError("campaign changed before submission intent")
            lineage = _lineage(plan._profile.target, plan._profile.resources)
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
                    "profile_label": plan._profile.label,
                    "retry_task_keys": [
                        key for key in allocation.allocation.task_keys if key in plan._retry
                    ],
                    "duplicate_risk_task_keys": [
                        key
                        for key in allocation.allocation.task_keys
                        if key in plan._duplicate_risk
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
                "outcome_revision": expected_revision + 1,
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


def _verify_view(state: dict[str, object], view: CampaignView) -> None:
    if view.campaign_id != state["campaign_id"]:
        raise PlanError("view belongs to another campaign")
    if view.revision != state["revision"]:
        raise PlanError("view is stale")
    tasks = _tasks_from_state(state)
    if tuple(item.key for item in view.tasks) != tuple(task.key for task in tasks):
        raise PlanError("view roster projection is invalid")
    if view.sealed != (state["phase"] == _SEALED):
        raise PlanError("view roster projection is invalid")

    attempts = cast(list[dict[str, object]], state["attempts"])
    if len(view.attempts) != len(attempts):
        raise PlanError("view attempt projection is invalid")
    current: dict[str, AttemptEvidence] = {}
    ambiguous: set[str] = set()
    for evidence, attempt in zip(view.attempts, attempts, strict=True):
        acceptance = cast(dict[str, object], attempt["acceptance"])
        status = AcceptanceState(cast(str, acceptance["status"]))
        receipt = None
        if status is AcceptanceState.ACCEPTED:
            receipt = JobReceipt(
                cast(str, attempt["allocation_id"]),
                cast(int, acceptance["job_id"]),
                cast(str | None, acceptance["cluster"]),
                tuple(cast(list[str], attempt["task_keys"])),
            )
        if (
            evidence.allocation_id != attempt["allocation_id"]
            or evidence.task_keys != tuple(cast(list[str], attempt["task_keys"]))
            or evidence.retry_task_keys != tuple(cast(list[str], attempt["retry_task_keys"]))
            or evidence.duplicate_risk_task_keys
            != tuple(cast(list[str], attempt["duplicate_risk_task_keys"]))
            or evidence.profile_label != attempt["profile_label"]
            or evidence.acceptance is not status
            or evidence.receipt != receipt
        ):
            raise PlanError("view attempt projection is invalid")
        if status is AcceptanceState.ACCEPTED:
            if view.scheduler_observed and evidence.allocation is None:
                raise PlanError("scheduler-observed view lacks accepted-attempt evidence")
        elif evidence.allocation is not None:
            raise PlanError("view contains scheduler evidence for an unaccepted attempt")
        if status is AcceptanceState.ACCEPTED:
            for key in evidence.task_keys:
                current[key] = evidence
        elif status is AcceptanceState.UNRESOLVED:
            ambiguous.update(evidence.task_keys)
            for key in evidence.task_keys:
                current[key] = evidence

    for task in view.tasks:
        current_attempt = current.get(task.key)
        expected_execution = (
            None
            if current_attempt is None or current_attempt.allocation is None
            else current_attempt.allocation.state
        )
        if (
            task.current_attempt_id
            != (None if current_attempt is None else current_attempt.allocation_id)
            or task.execution is not expected_execution
            or task.acceptance_ambiguous != (task.key in ambiguous)
            or (task.result is ResultState.UNOBSERVED) != (task.result_observed_at is None)
        ):
            raise PlanError("view task projection is invalid")

    if (
        any(attempt.acceptance is AcceptanceState.ACCEPTED for attempt in view.attempts)
        and not view.scheduler_observed
    ):
        raise PlanError("planning an accepted campaign requires scheduler-observed evidence")
    expected_ready = view.sealed and all(task.result is ResultState.VALID for task in view.tasks)
    terminal = {
        AllocationState.SUCCEEDED,
        AllocationState.FAILED,
        AllocationState.CANCELLED,
    }
    expected_quiescent = (
        view.scheduler_observed
        and not ambiguous
        and all(
            attempt.allocation is not None and attempt.allocation.state in terminal
            for attempt in view.attempts
            if attempt.acceptance is AcceptanceState.ACCEPTED
        )
    )
    if view.results_ready != expected_ready or view.quiescent != expected_quiescent:
        raise PlanError("view readiness projection is invalid")


def _select_tasks(
    view: CampaignView,
    retry: tuple[str, ...],
    duplicate_risk: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    retries = set(retry)
    acknowledged = set(duplicate_risk)
    active = {AllocationState.QUEUED, AllocationState.RUNNING}
    selected: list[str] = []
    excluded: list[str] = []
    used_acknowledgements: set[str] = set()
    accepted_by_task: dict[str, list[AttemptEvidence]] = {task.key: [] for task in view.tasks}
    for attempt in view.attempts:
        if attempt.acceptance is AcceptanceState.ACCEPTED:
            for key in attempt.task_keys:
                accepted_by_task[key].append(attempt)

    for task in view.tasks:
        if task.result is ResultState.VALID:
            if task.key in retries:
                raise PlanError(f"valid task {task.key!r} cannot be retried")
            excluded.append(task.key)
            continue
        if task.acceptance_ambiguous:
            if task.key in retries:
                raise PlanError(f"ambiguous task {task.key!r} cannot be retried")
            excluded.append(task.key)
            continue

        accepted = accepted_by_task[task.key]
        if not accepted:
            if task.key in retries:
                raise PlanError("retry requires an earlier accepted attempt")
            selected.append(task.key)
            continue

        states = tuple(
            attempt.allocation.state for attempt in accepted if attempt.allocation is not None
        )
        if len(states) != len(accepted):
            raise PlanError("accepted attempt lacks scheduler evidence")
        if any(state in active for state in states):
            if task.key in retries:
                raise PlanError(f"active task {task.key!r} cannot be retried")
            excluded.append(task.key)
            continue
        if task.key not in retries:
            excluded.append(task.key)
            continue
        if AllocationState.UNKNOWN in states:
            if task.key not in acknowledged:
                raise PlanError(
                    f"unknown task {task.key!r} retry requires duplicate-risk acknowledgement"
                )
            used_acknowledgements.add(task.key)
        elif task.key in acknowledged:
            raise PlanError(
                f"duplicate-risk acknowledgement for {task.key!r} has no unknown attempt"
            )
        selected.append(task.key)

    unused = acknowledged - used_acknowledgements
    if unused:
        raise PlanError("duplicate-risk acknowledgement does not match unknown accepted work")
    return tuple(selected), tuple(excluded)


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


def _attempt_queries(state: dict[str, object]) -> tuple[_slurm._AttemptQuery, ...]:
    values: list[_slurm._AttemptQuery] = []
    for attempt in cast(list[dict[str, object]], state["attempts"]):
        acceptance = cast(dict[str, object], attempt["acceptance"])
        if acceptance["status"] != _ACCEPTED:
            continue
        values.append(
            _slurm._AttemptQuery(
                cast(str, attempt["allocation_id"]),
                cast(int, acceptance["job_id"]),
                cast(str | None, acceptance["cluster"]),
                cast(str, attempt["window_start"]),
                cast(str, attempt["window_end"]),
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
    outcome_count = 0
    for raw in value:
        if not isinstance(raw, dict):
            continue
        acceptance = cast(dict[str, object], raw).get("acceptance")
        if isinstance(acceptance, dict) and cast(dict[str, object], acceptance).get("status") in {
            _ACCEPTED,
            _NOT_SUBMITTED,
        }:
            outcome_count += 1
    mutation_count = len(value) + outcome_count
    if not (
        mutation_count + int(sealed)
        <= state_revision
        <= mutation_count + len(task_keys) + int(sealed)
    ):
        raise TaskConflict("campaign revision cannot be explained by durable history")
    expected = {
        "allocation_id",
        "task_keys",
        "campaign_revision",
        "profile_label",
        "retry_task_keys",
        "duplicate_risk_task_keys",
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
    accepted_outcome_by_task: dict[str, int] = {}
    unresolved_until_by_task: dict[str, int | None] = {}
    mutation_revisions: set[int] = set()
    plan_requirements: list[tuple[int, int]] = []
    task_positions = {key: index for index, key in enumerate(task_keys)}
    plan_revisions: dict[str, int] = {}
    group_revision: int | None = None
    group_plan_digest: str | None = None
    group_profile_label: str | None = None
    group_last_task_position = -1
    previous_intent_revision = -1
    previous_status: str | None = None
    previous_outcome_revision: int | None = None
    if not value:
        if lineage is not None:
            raise TaskConflict("campaign lineage has no attempt history")
        _validate_revision_history(state_revision, set(), [], len(task_keys), sealed=sealed)
        return
    if lineage is None:
        raise TaskConflict("campaign attempt has no resource lineage")
    target, resources = lineage
    lineage_values = _lineage(target, resources)
    for raw in value:
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
        profile_label = attempt["profile_label"]
        if not isinstance(profile_label, str) or not profile_label:
            raise TaskConflict("campaign attempt profile label is invalid")
        known_revision = plan_revisions.setdefault(plan_digest, campaign_revision)
        if known_revision != campaign_revision:
            raise TaskConflict("campaign attempt plan revision is inconsistent")
        new_group = group_revision is None or campaign_revision != group_revision
        if new_group:
            group_revision = campaign_revision
            group_plan_digest = plan_digest
            group_profile_label = profile_label
            group_last_task_position = -1
            plan_requirements.append((campaign_revision, 0))
        else:
            if plan_digest != group_plan_digest or profile_label != group_profile_label:
                raise TaskConflict("campaign attempt plan group is invalid")
        positions = tuple(task_positions[key] for key in typed_attempt_keys)
        if positions[0] <= group_last_task_position:
            raise TaskConflict("campaign plan allocation sequence is invalid")
        group_last_task_position = positions[-1]
        plan_revision, required_prefix = plan_requirements[-1]
        plan_requirements[-1] = (plan_revision, max(required_prefix, positions[-1] + 1))
        if new_group:
            intent_revision = campaign_revision + 1
            if intent_revision <= previous_intent_revision:
                raise TaskConflict("campaign attempt revision precedes durable history")
        else:
            if (
                previous_status != _ACCEPTED
                or previous_outcome_revision != previous_intent_revision + 1
            ):
                raise TaskConflict("campaign plan outcome is terminal")
            assert previous_outcome_revision is not None
            intent_revision = previous_outcome_revision + 1
        if intent_revision > state_revision or intent_revision in mutation_revisions:
            raise TaskConflict("campaign attempt intent revision is invalid")
        mutation_revisions.add(intent_revision)
        for key in typed_attempt_keys:
            unresolved_until = unresolved_until_by_task.get(key, 0)
            if unresolved_until is None or unresolved_until > intent_revision:
                raise TaskConflict("campaign attempt overlaps unresolved history")
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
        expected_retries = tuple(
            key
            for key in typed_attempt_keys
            if accepted_outcome_by_task.get(key, state_revision + 1) < intent_revision
        )
        if typed_retries != expected_retries:
            raise TaskConflict("campaign attempt retry keys do not match accepted history")
        duplicate_risk_keys = attempt["duplicate_risk_task_keys"]
        if not isinstance(duplicate_risk_keys, list):
            raise TaskConflict("campaign attempt duplicate-risk keys are invalid")
        typed_duplicate_risk = tuple(cast(list[object], duplicate_risk_keys))
        if (
            any(not isinstance(key, str) for key in typed_duplicate_risk)
            or len(set(typed_duplicate_risk)) != len(typed_duplicate_risk)
            or tuple(key for key in typed_retries if key in set(typed_duplicate_risk))
            != typed_duplicate_risk
        ):
            raise TaskConflict("campaign attempt duplicate-risk keys are invalid")
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
        outcome_revision: int | None = None
        if status == _UNRESOLVED:
            if set(typed_acceptance) != {"status"}:
                raise TaskConflict("campaign attempt acceptance is invalid")
        elif status in {_ACCEPTED, _NOT_SUBMITTED}:
            raw_outcome_revision = typed_acceptance.get("outcome_revision")
            job_id = typed_acceptance.get("job_id")
            cluster = typed_acceptance.get("cluster")
            expected_acceptance = (
                {"status", "outcome_revision"}
                if status == _NOT_SUBMITTED
                else {"status", "job_id", "cluster", "outcome_revision"}
            )
            invalid_receipt = status == _ACCEPTED and (
                type(job_id) is not int
                or job_id <= 0
                or (
                    cluster is not None
                    and (not isinstance(cluster, str) or _TOKEN.fullmatch(cluster) is None)
                )
            )
            if set(typed_acceptance) != expected_acceptance or invalid_receipt:
                raise TaskConflict("campaign attempt acceptance is invalid")
            if (
                type(raw_outcome_revision) is not int
                or raw_outcome_revision <= intent_revision
                or raw_outcome_revision > state_revision
                or raw_outcome_revision in mutation_revisions
            ):
                raise TaskConflict("campaign attempt outcome revision is invalid")
            outcome_revision = raw_outcome_revision
            mutation_revisions.add(outcome_revision)
            if status == _ACCEPTED:
                for key in typed_attempt_keys:
                    accepted_outcome_by_task[key] = min(
                        outcome_revision,
                        accepted_outcome_by_task.get(key, outcome_revision),
                    )
        else:
            raise TaskConflict("campaign attempt acceptance is invalid")
        for key in typed_attempt_keys:
            unresolved_until_by_task[key] = outcome_revision
        allocation_ids.add(allocation_id)
        previous_intent_revision = intent_revision
        previous_status = cast(str, status)
        previous_outcome_revision = outcome_revision
    _validate_revision_history(
        state_revision,
        mutation_revisions,
        plan_requirements,
        len(task_keys),
        sealed=sealed,
    )


def _plan_strings(value: object, *, name: str) -> list[str]:
    if not isinstance(value, list):
        raise PlanError(f"plan {name} must be an array of strings")
    strings: list[str] = []
    for item in cast(list[object], value):
        if not isinstance(item, str) or "\0" in item:
            raise PlanError(f"plan {name} must be an array of strings")
        strings.append(item)
    return strings


def _plan_timestamp(value: object, *, name: str, optional: bool = False) -> datetime | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise PlanError(f"plan {name} is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise PlanError(f"plan {name} is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0) or parsed.isoformat() != value:
        raise PlanError(f"plan {name} is invalid")
    return parsed


def _optional_plan_string(value: object, *, name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise PlanError(f"plan {name} is invalid")
    return value


def _plan_string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise PlanError(f"plan {name} is invalid")
    return value


def _plan_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise PlanError(f"plan {name} is invalid")
    return value


def _plan_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PlanError(f"plan {name} is invalid")
    return value


def _view_from_dict(value: object) -> CampaignView:
    if not isinstance(value, dict):
        raise PlanError("plan view is invalid")
    mapping = cast(dict[str, object], value)
    if set(mapping) != {
        "campaign_id",
        "revision",
        "sealed",
        "tasks",
        "attempts",
        "scheduler_observed",
        "observed_at",
        "results_ready",
        "quiescent",
    }:
        raise PlanError("plan view is invalid")
    raw_tasks = mapping["tasks"]
    raw_attempts = mapping["attempts"]
    if not isinstance(raw_tasks, list) or not isinstance(raw_attempts, list):
        raise PlanError("plan view is invalid")
    tasks: list[TaskEvidence] = []
    attempts: list[AttemptEvidence] = []
    try:
        for raw in cast(list[object], raw_tasks):
            if not isinstance(raw, dict):
                raise PlanError("plan task evidence is invalid")
            item = cast(dict[str, object], raw)
            if set(item) != {
                "key",
                "result",
                "result_observed_at",
                "current_attempt_id",
                "execution",
                "acceptance_ambiguous",
            }:
                raise PlanError("plan task evidence is invalid")
            execution = item["execution"]
            tasks.append(
                TaskEvidence(
                    _plan_string(item["key"], name="task key"),
                    ResultState(_plan_string(item["result"], name="result state")),
                    _plan_timestamp(
                        item["result_observed_at"],
                        name="result observation time",
                        optional=True,
                    ),
                    _optional_plan_string(
                        item["current_attempt_id"], name="current attempt identity"
                    ),
                    None
                    if execution is None
                    else AllocationState(_plan_string(execution, name="execution state")),
                    _plan_bool(item["acceptance_ambiguous"], name="acceptance ambiguity"),
                )
            )
        for raw in cast(list[object], raw_attempts):
            if not isinstance(raw, dict):
                raise PlanError("plan attempt evidence is invalid")
            item = cast(dict[str, object], raw)
            if set(item) != {
                "allocation_id",
                "task_keys",
                "retry_task_keys",
                "duplicate_risk_task_keys",
                "profile_label",
                "acceptance",
                "receipt",
                "allocation",
            }:
                raise PlanError("plan attempt evidence is invalid")
            allocation_id = _plan_string(item["allocation_id"], name="allocation identity")
            task_keys = tuple(_plan_strings(item["task_keys"], name="attempt task keys"))
            receipt = None
            raw_receipt = item["receipt"]
            if raw_receipt is not None:
                if not isinstance(raw_receipt, dict):
                    raise PlanError("plan receipt evidence is invalid")
                receipt_values = cast(dict[str, object], raw_receipt)
                if set(receipt_values) != {"allocation_id", "job_id", "cluster", "task_keys"}:
                    raise PlanError("plan receipt evidence is invalid")
                receipt = JobReceipt(
                    _plan_string(receipt_values["allocation_id"], name="receipt allocation"),
                    _plan_integer(receipt_values["job_id"], name="receipt job", minimum=1),
                    _optional_plan_string(receipt_values["cluster"], name="receipt cluster"),
                    tuple(_plan_strings(receipt_values["task_keys"], name="receipt task keys")),
                )
            allocation = None
            raw_allocation = item["allocation"]
            if raw_allocation is not None:
                if not isinstance(raw_allocation, dict):
                    raise PlanError("plan allocation evidence is invalid")
                allocation_values = cast(dict[str, object], raw_allocation)
                if set(allocation_values) != {
                    "state",
                    "raw_state",
                    "accounting_state",
                    "exit_code",
                    "reason",
                    "started_at",
                    "ended_at",
                    "observed_at",
                }:
                    raise PlanError("plan allocation evidence is invalid")
                observed = _plan_timestamp(
                    allocation_values["observed_at"], name="allocation observation time"
                )
                assert observed is not None
                allocation = AllocationEvidence(
                    AllocationState(
                        _plan_string(allocation_values["state"], name="allocation state")
                    ),
                    _optional_plan_string(allocation_values["raw_state"], name="raw state"),
                    _optional_plan_string(
                        allocation_values["accounting_state"], name="accounting state"
                    ),
                    _optional_plan_string(allocation_values["exit_code"], name="exit code"),
                    _optional_plan_string(allocation_values["reason"], name="reason"),
                    _optional_plan_string(allocation_values["started_at"], name="start time"),
                    _optional_plan_string(allocation_values["ended_at"], name="end time"),
                    observed,
                )
            attempts.append(
                AttemptEvidence(
                    allocation_id,
                    task_keys,
                    tuple(_plan_strings(item["retry_task_keys"], name="attempt retry keys")),
                    tuple(
                        _plan_strings(
                            item["duplicate_risk_task_keys"],
                            name="attempt duplicate-risk keys",
                        )
                    ),
                    _plan_string(item["profile_label"], name="attempt profile label"),
                    AcceptanceState(_plan_string(item["acceptance"], name="attempt acceptance")),
                    receipt,
                    allocation,
                )
            )
        observed_at = _plan_timestamp(mapping["observed_at"], name="view observation time")
        assert observed_at is not None
        view = CampaignView(
            _plan_string(mapping["campaign_id"], name="view campaign identity"),
            _plan_integer(mapping["revision"], name="view revision"),
            _plan_bool(mapping["sealed"], name="view sealed state"),
            tuple(tasks),
            tuple(attempts),
            _plan_bool(mapping["scheduler_observed"], name="scheduler observation state"),
            observed_at,
            _plan_bool(mapping["results_ready"], name="result readiness"),
            _plan_bool(mapping["quiescent"], name="quiescence"),
        )
    except (TypeError, ValueError) as error:
        raise PlanError("plan view is invalid") from error
    if _view_semantics(view) != mapping:
        raise PlanError("plan view values are invalid")
    return view


def _plan_inputs(
    document: object,
) -> tuple[
    dict[str, object],
    Profile,
    CampaignView,
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
        "profile_label",
        "target",
        "resources",
        "roster_digest",
        "view",
        "view_digest",
        "selected",
        "excluded",
        "retry",
        "duplicate_risk",
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
    label = mapping["profile_label"]
    if not isinstance(label, str):
        raise PlanError("plan profile label is invalid")
    try:
        profile = Profile(
            label,
            _target_from_dict(mapping["target"]),
            _resource_from_dict(mapping["resources"]),
        )
    except ConfigurationError as error:
        raise PlanError("plan profile is invalid") from error
    view = _view_from_dict(mapping["view"])
    retry = _plan_strings(mapping["retry"], name="retry")
    duplicate_risk = _plan_strings(mapping["duplicate_risk"], name="duplicate_risk")
    for name in ("selected", "excluded"):
        _plan_strings(mapping[name], name=name)
    if len(set(retry)) != len(retry) or len(set(duplicate_risk)) != len(duplicate_risk):
        raise PlanError("plan selections contain duplicate task keys")
    tasks_per = mapping["tasks_per_allocation"]
    if tasks_per is not None and (
        isinstance(tasks_per, bool) or not isinstance(tasks_per, int) or tasks_per < 1
    ):
        raise PlanError("plan tasks_per_allocation must be an integer >= 1")
    return mapping, profile, view, retry, duplicate_risk, tasks_per


def restore_plan(campaign: Campaign, document: object) -> SubmissionPlan:
    mapping, profile, view, retry, duplicate_risk, tasks_per = _plan_inputs(document)
    plan = campaign.plan(
        profile,
        view=view,
        retry=retry,
        allow_duplicate_risk=duplicate_risk,
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
        result = _slurm._run_ssh(plan._profile.target, (*item.argv, "--test-only"), item.script)
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
