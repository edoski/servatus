# pyright: reportUnnecessaryIsInstance=false, reportPrivateUsage=false
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import ClassVar, Self, cast

from ._errors import ConfigurationError
from ._slurm import AllocationState

SCHEMA_VERSION = 5

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_GRES = re.compile(r"gpu(?::[A-Za-z0-9][A-Za-z0-9._-]*)?\Z")
_DURATION = re.compile(r"(?:(0|[1-9][0-9]*)-)?([0-9]{2}):([0-5][0-9]):([0-5][0-9])\Z")


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


def effective_time_limit(value: str) -> str:
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
    if _CONTROL.search(raw) or not path.is_absolute() or ".." in path.parts:
        raise ConfigurationError(f"{name} must be an absolute POSIX path without parent traversal")
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
    args: Sequence[str]
    stdin: bytes = b""

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key or "\0" in self.key:
            raise ConfigurationError("Task.key must be a nonempty string without NUL")
        if isinstance(self.args, (str, bytes, bytearray)) or not isinstance(self.args, Sequence):
            raise ConfigurationError("Task.args must be a sequence of strings")
        args = tuple(self.args)
        if any(not isinstance(arg, str) for arg in args):
            raise ConfigurationError("Task.args must be a sequence of strings")
        object.__setattr__(self, "args", args)
        if any("\0" in arg for arg in self.args):
            raise ConfigurationError("Task.args cannot contain NUL")
        if not isinstance(self.stdin, bytes):
            raise ConfigurationError("Task.stdin must be bytes")


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceRequest:
    cpus_per_task: int
    memory_mib_per_task: int
    gpus_per_task: int = 0
    time_limit: str

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {"cpus_per_task", "memory_mib_per_task", "gpus_per_task", "time_limit"}
    )

    def __post_init__(self) -> None:
        _integer(self.cpus_per_task, minimum=1, name="cpus_per_task")
        _integer(self.memory_mib_per_task, minimum=1, name="memory_mib_per_task")
        _integer(self.gpus_per_task, minimum=0, name="gpus_per_task")
        _duration_seconds(self.time_limit, name="time_limit")


@dataclass(frozen=True, slots=True, kw_only=True)
class SlurmTarget:
    host: str
    slurm_bin: PurePosixPath
    apptainer: PurePosixPath
    image: PurePosixPath
    work_root: PurePosixPath
    log_root: PurePosixPath
    partitions: Sequence[str]
    account: str | None = None
    qos: str | None = None
    constraint: str | None = None
    gpu_gres: str | None = None
    max_tasks_per_allocation: int
    max_cpus_per_allocation: int
    max_memory_mib_per_allocation: int
    max_gpus_per_allocation: int
    max_time_limit: str
    max_allocations_per_submit: int = 1
    max_script_bytes: int = 1024 * 1024

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
        }
    )
    _OPTIONAL: ClassVar[frozenset[str]] = frozenset(
        {
            "account",
            "qos",
            "constraint",
            "gpu_gres",
            "max_allocations_per_submit",
            "max_script_bytes",
        }
    )

    def __post_init__(self) -> None:
        _safe_token(self.host, name="host")
        for name in ("slurm_bin", "apptainer", "image", "work_root", "log_root"):
            object.__setattr__(self, name, _absolute_path(getattr(self, name), name=name))
        _configuration(
            isinstance(self.partitions, Sequence)
            and not isinstance(self.partitions, str)
            and bool(self.partitions),
            "partitions must be a nonempty sequence of strings",
        )
        object.__setattr__(self, "partitions", tuple(self.partitions))
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
        gpus_per_task=cast(int, mapping.get("gpus_per_task", 0)),
        time_limit=cast(str, mapping["time_limit"]),
    )


def _profile_resources(values: object) -> ResourceRequest:
    if not isinstance(values, dict):
        raise ConfigurationError("resources must be a table")
    mapping = cast(dict[str, object], values)
    unknown = mapping.keys() - ResourceRequest._KEYS
    missing = ResourceRequest._KEYS - {"gpus_per_task"} - mapping.keys()
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
        max_allocations_per_submit=cast(int, mapping.get("max_allocations_per_submit", 1)),
        max_script_bytes=cast(int, mapping.get("max_script_bytes", 1024 * 1024)),
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

        tables: dict[str, dict[str, object]] = {}
        for label, raw in cast(dict[str, object], profiles).items():
            _nonempty_string(label, name="profile label")
            if not isinstance(raw, dict):
                raise ConfigurationError(f"profile {label!r} must be a table")
            mapping = cast(dict[str, object], raw)
            if mapping.keys() - {"target", "resources"}:
                raise ConfigurationError(f"unknown profile keys in {label!r}")
            for section, allowed in (
                ("target", SlurmTarget._REQUIRED | SlurmTarget._OPTIONAL),
                ("resources", ResourceRequest._KEYS),
            ):
                table = mapping.get(section)
                if isinstance(table, dict) and cast(dict[str, object], table).keys() - allowed:
                    raise ConfigurationError(f"unknown {section} keys in {label!r}")
            tables[label] = mapping
        default = document.get("default_profile")
        if default is not None:
            _nonempty_string(default, name="default_profile")
            if default not in tables:
                raise ConfigurationError("default_profile does not name a declared profile")
        selected = name if name is not None else cast(str | None, default)
        if selected is None and len(tables) == 1:
            selected = next(iter(tables))
        if selected is None:
            raise ConfigurationError("profile selection is required")
        _nonempty_string(selected, name="profile name")
        if selected not in tables:
            raise ConfigurationError(f"profile {selected!r} is not declared")
        mapping = tables[selected]
        return cls(
            selected,
            _profile_target(mapping.get("target")),
            _profile_resources(mapping.get("resources")),
        )


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


class AcceptanceState(StrEnum):
    UNRESOLVED = "UNRESOLVED"
    ACCEPTED = "ACCEPTED"
    NOT_SUBMITTED = "NOT_SUBMITTED"


class ResultState(StrEnum):
    UNOBSERVED = "UNOBSERVED"
    MISSING = "MISSING"
    VALID = "VALID"


ResultProbe = Callable[[Task], bool]


@dataclass(frozen=True, slots=True)
class JobReceipt:
    allocation_id: str
    job_id: int
    cluster: str | None
    task_keys: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.job_id}" + (f";{self.cluster}" if self.cluster else "")


@dataclass(frozen=True, slots=True)
class RegisteredTask:
    task: Task
    revision: int


@dataclass(frozen=True, slots=True)
class Attempt:
    allocation_id: str
    task_keys: tuple[str, ...]
    profile: Profile
    retry_task_keys: tuple[str, ...]
    duplicate_risk_task_keys: tuple[str, ...]
    intent_revision: int
    intent_at: datetime
    acceptance: AcceptanceState = AcceptanceState.UNRESOLVED
    receipt: JobReceipt | None = None
    outcome_revision: int | None = None

    @property
    def window_start(self) -> str:
        return (self.intent_at - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")

    @property
    def window_end(self) -> str:
        return (self.intent_at + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")


@dataclass(frozen=True, slots=True)
class State:
    campaign_id: str
    revision: int
    roster: tuple[RegisteredTask, ...]
    sealed_revision: int | None
    attempts: tuple[Attempt, ...] = ()

    @property
    def tasks(self) -> tuple[Task, ...]:
        return tuple(item.task for item in self.roster)

    @property
    def sealed(self) -> bool:
        return self.sealed_revision is not None


@dataclass(frozen=True, slots=True)
class PlannedAllocation:
    allocation_id: str
    task_keys: tuple[str, ...]
    cpus: int
    memory_mib: int
    gpus: int
    time_limit: str
    script: bytes = field(repr=False)
    argv: tuple[str, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class SubmissionPlan:
    campaign_id: str
    revision: int
    profile: Profile
    selected_task_keys: tuple[str, ...]
    excluded_task_keys: tuple[str, ...]
    deferred_task_keys: tuple[str, ...]
    retry_task_keys: tuple[str, ...]
    duplicate_risk_task_keys: tuple[str, ...]
    tasks_per_allocation: int
    probe_required: bool
    allocations: tuple[PlannedAllocation, ...]
    digest: str

    @property
    def warnings(self) -> tuple[str, ...]:
        if not self.duplicate_risk_task_keys:
            return ()
        return (
            "duplicate execution risk accepted for retries: "
            + ", ".join(self.duplicate_risk_task_keys),
        )


@dataclass(frozen=True, slots=True)
class UnresolvedSubmission:
    allocation_id: str
    task_keys: tuple[str, ...]
    observed_receipt: JobReceipt | None = None


@dataclass(frozen=True, slots=True)
class SubmitResult:
    receipts: tuple[JobReceipt, ...]
    unresolved: tuple[UnresolvedSubmission, ...]
    unattempted: tuple[PlannedAllocation, ...]
    stop_reason: str | None


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
    retained: bool = False


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

    def to_json(self) -> bytes:
        """Serialize diagnostic evidence without execution configuration or task payloads."""
        value = asdict(self)
        value["observed_at"] = self.observed_at.isoformat()
        for task in value["tasks"]:
            at = task["result_observed_at"]
            task["result_observed_at"] = None if at is None else at.isoformat()
        for attempt in value["attempts"]:
            allocation = attempt["allocation"]
            if allocation is not None:
                allocation["observed_at"] = allocation["observed_at"].isoformat()
        return canonical(value)


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
    controller_stdout: str
    controller_stderr: str


def decode_json(data: bytes) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    return json.loads(data, object_pairs_hook=pairs)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def object_fields(value: object, fields: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or value.keys() != fields:
        raise ValueError("unexpected object fields")
    return cast(dict[str, object], value)


def integer(value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError("invalid integer")
    return value


def boolean(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError("invalid boolean")
    return value


def string(value: object) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError("invalid string")
    return value


def array(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("invalid array")
    return cast(list[object], value)


def keys(value: object) -> tuple[str, ...]:
    result = tuple(string(item) for item in array(value))
    if len(set(result)) != len(result):
        raise ValueError("duplicate keys")
    return result


def identifier(value: object, length: int) -> str:
    result = string(value)
    if re.fullmatch(f"[0-9a-f]{{{length}}}", result) is None:
        raise ValueError("invalid identifier")
    return result


def profile_document(profile: Profile) -> dict[str, object]:
    return {
        "label": profile.label,
        "target": _target_dict(profile.target),
        "resources": _resource_dict(profile.resources),
    }


def profile_from_document(value: object) -> Profile:
    obj = object_fields(value, {"label", "target", "resources"})
    target = object_fields(obj["target"], set(SlurmTarget._REQUIRED | SlurmTarget._OPTIONAL))
    resources = object_fields(obj["resources"], set(ResourceRequest._KEYS))
    return Profile(
        cast(str, obj["label"]), _target_from_values(target), _resource_from_values(resources)
    )


def receipt_document(receipt: JobReceipt) -> dict[str, object]:
    return {
        "allocation_id": receipt.allocation_id,
        "job_id": receipt.job_id,
        "cluster": receipt.cluster,
        "task_keys": list(receipt.task_keys),
    }


def state_document(state: State) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": state.campaign_id,
        "revision": state.revision,
        "sealed_revision": state.sealed_revision,
        "tasks": [
            {
                "key": item.task.key,
                "args": list(item.task.args),
                "stdin": base64.b64encode(item.task.stdin).decode("ascii"),
                "revision": item.revision,
            }
            for item in state.roster
        ],
        "attempts": [
            {
                "allocation_id": attempt.allocation_id,
                "task_keys": list(attempt.task_keys),
                "profile": profile_document(attempt.profile),
                "retry_task_keys": list(attempt.retry_task_keys),
                "duplicate_risk_task_keys": list(attempt.duplicate_risk_task_keys),
                "intent_revision": attempt.intent_revision,
                "intent_at": attempt.intent_at.isoformat(),
                "acceptance": attempt.acceptance.value,
                "outcome_revision": attempt.outcome_revision,
                "receipt": None
                if attempt.receipt is None
                else {"job_id": attempt.receipt.job_id, "cluster": attempt.receipt.cluster},
            }
            for attempt in state.attempts
        ],
    }


def decode_state(value: object) -> State:
    obj = object_fields(
        value, {"schema_version", "campaign_id", "revision", "sealed_revision", "tasks", "attempts"}
    )
    if integer(obj["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("unsupported campaign schema")
    campaign_id = identifier(obj["campaign_id"], 32)
    revision = integer(obj["revision"])
    sealed = None if obj["sealed_revision"] is None else integer(obj["sealed_revision"])
    if sealed is not None and sealed > revision:
        raise ValueError("invalid seal revision")
    roster: list[RegisteredTask] = []
    for item in array(obj["tasks"]):
        raw = object_fields(item, {"key", "args", "stdin", "revision"})
        args = array(raw["args"])
        if any(not isinstance(arg, str) for arg in args):
            raise ValueError("invalid task arguments")
        if not isinstance(raw["stdin"], str):
            raise ValueError("invalid task stdin")
        try:
            payload = base64.b64decode(raw["stdin"], validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("invalid task stdin") from error
        task = Task(string(raw["key"]), tuple(cast(list[str], args)), payload)
        introduced = integer(raw["revision"])
        if introduced > revision or (roster and introduced < roster[-1].revision):
            raise ValueError("invalid task revision")
        if sealed is not None and introduced > sealed:
            raise ValueError("task appended after seal")
        roster.append(RegisteredTask(task, introduced))
    by_key = {entry.task.key: entry for entry in roster}
    if len(by_key) != len(roster):
        raise ValueError("duplicate task key")
    events = {entry.revision for entry in roster if entry.revision}
    if sealed:
        if sealed in events:
            raise ValueError("conflicting seal revision")
        events.add(sealed)
    attempts: list[Attempt] = []
    ids: set[str] = set()
    for item in array(obj["attempts"]):
        raw = object_fields(
            item,
            {
                "allocation_id",
                "task_keys",
                "profile",
                "retry_task_keys",
                "duplicate_risk_task_keys",
                "intent_revision",
                "intent_at",
                "acceptance",
                "receipt",
                "outcome_revision",
            },
        )
        allocation_id = identifier(raw["allocation_id"], 24)
        if allocation_id in ids:
            raise ValueError("duplicate allocation identity")
        ids.add(allocation_id)
        intent = integer(raw["intent_revision"], 1)
        if (
            intent > revision
            or intent in events
            or (attempts and intent <= attempts[-1].intent_revision)
        ):
            raise ValueError("invalid intent revision")
        events.add(intent)
        selected = keys(raw["task_keys"])
        if not selected or any(
            key not in by_key or by_key[key].revision >= intent for key in selected
        ):
            raise ValueError("invalid attempt task references")
        if tuple(key for key in by_key if key in selected) != selected:
            raise ValueError("attempt tasks are not in roster order")
        retry = keys(raw["retry_task_keys"])
        ack = keys(raw["duplicate_risk_task_keys"])
        if tuple(key for key in selected if key in retry) != retry:
            raise ValueError("invalid retry references")
        if tuple(key for key in retry if key in ack) != ack:
            raise ValueError("invalid duplicate-risk references")
        previous = {
            key
            for prior in attempts
            if prior.acceptance is AcceptanceState.ACCEPTED
            and prior.outcome_revision is not None
            and prior.outcome_revision < intent
            for key in prior.task_keys
        }
        if tuple(key for key in selected if key in previous) != retry:
            raise ValueError("retry choices disagree with prior accepted work")
        if any(
            set(prior.task_keys) & set(selected)
            for prior in attempts
            if prior.outcome_revision is None or prior.outcome_revision >= intent
        ):
            raise ValueError("attempt overlaps unresolved intent")
        at_text = string(raw["intent_at"])
        at = datetime.fromisoformat(at_text)
        if at.tzinfo is None or at.utcoffset() != timedelta(0) or at.isoformat() != at_text:
            raise ValueError("intent time must be canonical UTC")
        status = AcceptanceState(string(raw["acceptance"]))
        outcome = None if raw["outcome_revision"] is None else integer(raw["outcome_revision"], 1)
        receipt = None
        if status is AcceptanceState.UNRESOLVED:
            if outcome is not None or raw["receipt"] is not None:
                raise ValueError("unresolved intent has outcome")
        else:
            if outcome is None or not intent < outcome <= revision or outcome in events:
                raise ValueError("invalid outcome revision")
            events.add(outcome)
            if status is AcceptanceState.ACCEPTED:
                accepted = object_fields(raw["receipt"], {"job_id", "cluster"})
                cluster = _safe_token(accepted["cluster"], name="cluster", optional=True)
                receipt = JobReceipt(
                    allocation_id, integer(accepted["job_id"], 1), cluster, selected
                )
            elif raw["receipt"] is not None:
                raise ValueError("not-submitted outcome has receipt")
        profile = profile_from_document(raw["profile"])
        target, resources = profile.target, profile.resources
        if (
            len(selected) > target.max_tasks_per_allocation
            or len(selected) * resources.cpus_per_task > target.max_cpus_per_allocation
            or len(selected) * resources.memory_mib_per_task > target.max_memory_mib_per_allocation
            or len(selected) * resources.gpus_per_task > target.max_gpus_per_allocation
            or _duration_seconds(effective_time_limit(resources.time_limit), name="time_limit")
            > _duration_seconds(effective_time_limit(target.max_time_limit), name="max_time_limit")
        ):
            raise ValueError("attempt exceeds its target capacity")
        attempts.append(
            Attempt(
                allocation_id,
                selected,
                profile,
                retry,
                ack,
                intent,
                at,
                status,
                receipt,
                outcome,
            )
        )
    if revision != max(events, default=0):
        raise ValueError("revision does not match recorded mutations")
    return State(campaign_id, revision, tuple(roster), sealed, tuple(attempts))
