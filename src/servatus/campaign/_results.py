"""Results returned by Campaign operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from . import _codec


def to_document(value: object) -> object:
    """JSON-compatible data for a campaign value (a result, receipt, check, or planned
    allocation), as the strict codec writes it: durations as ``[D-]HH:MM:SS`` text, bytes as
    base64, enums by value, and nested dataclasses as objects."""
    return _codec.dump(value)


@dataclass(frozen=True, slots=True)
class Receipt:
    """Slurm accepted this allocation. Acceptance is not completion."""

    allocation_id: str
    job: int
    task_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UnresolvedSubmission:
    """Durable intent exists, but acceptance was not durably recorded.

    ``observed_job`` is set when Slurm answered but recording the receipt failed. ``reconcile``
    records it from Slurm's evidence.
    """

    allocation_id: str
    task_keys: tuple[str, ...]
    observed_job: int | None = None


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
class WaitingAllocation:
    """An unresolved allocation without evidence yet. ``reconcile`` records it as not submitted
    from ``decidable_at`` on, if Slurm still shows nothing for it."""

    allocation_id: str
    task_keys: tuple[str, ...]
    decidable_at: datetime


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    """What one ``reconcile`` recorded: receipts proven by Slurm's evidence, allocations recorded
    as not submitted, and allocations still ``waiting`` for evidence or time."""

    receipts: tuple[Receipt, ...] = ()
    not_submitted: tuple[UnattemptedAllocation, ...] = ()
    waiting: tuple[WaitingAllocation, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.waiting


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
