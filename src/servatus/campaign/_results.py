"""Results returned by Campaign operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from . import _codec
from ._evidence import JobRef


def to_document(value: object) -> object:
    """JSON-compatible data for a campaign value (a result, receipt, check, or planned
    allocation), as the strict codec writes it: durations as ``[D-]HH:MM:SS`` text, bytes as
    base64, enums by value, and nested values such as ``JobRef`` as objects."""
    return _codec.dump(value)


@dataclass(frozen=True, slots=True)
class Receipt:
    """Slurm accepted this allocation. Acceptance is not completion."""

    allocation_id: str
    job: JobRef
    task_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UnresolvedSubmission:
    """Durable intent exists, but acceptance was not durably recorded.

    ``observed_job`` is set when Slurm answered but recording the receipt failed. Save it and
    reconcile before retrying.
    """

    allocation_id: str
    task_keys: tuple[str, ...]
    observed_job: JobRef | None = None


@dataclass(frozen=True, slots=True)
class UnattemptedAllocation:
    allocation_id: str
    task_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SubmitResult:
    receipts: tuple[Receipt, ...]
    unresolved: tuple[UnresolvedSubmission, ...] = ()
    unattempted: tuple[UnattemptedAllocation, ...] = ()
    stop_reason: str | None = None

    @property
    def complete(self) -> bool:
        return self.stop_reason is None


@dataclass(frozen=True, slots=True)
class ShapeCheck:
    """One time-specific ``sbatch --test-only`` answer for a distinct allocation shape."""

    task_count: int
    cpus: int
    memory_mib: int
    gpus: int
    time_limit: timedelta
    accepted: bool
    scheduler_stdout: str
    scheduler_stderr: str


@dataclass(frozen=True, slots=True)
class LogSnapshot:
    """A bounded binary suffix of one allocation or Task log. Sensitive and untrusted."""

    content: bytes = field(repr=False)
    truncated: bool
    observed_at: datetime
