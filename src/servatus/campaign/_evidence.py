"""Scheduler evidence: normalized states, job identities, and pure Slurm output parsing.

The type definitions at the top are the shared vocabulary of the campaign layer. Parsing and
normalization functions below them are pure; ``_scheduler`` performs the remote calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..errors import ConfigurationError
from ._config import TOKEN


class AllocationState(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"

    @property
    def active(self) -> bool:
        return self in (AllocationState.QUEUED, AllocationState.RUNNING)

    @property
    def terminal(self) -> bool:
        return self in (
            AllocationState.SUCCEEDED,
            AllocationState.FAILED,
            AllocationState.CANCELLED,
        )


@dataclass(frozen=True, slots=True)
class JobRef:
    """A positive Slurm job number and optional federation cluster name."""

    job_id: int
    cluster: str | None = None

    def __post_init__(self) -> None:
        if type(self.job_id) is not int or self.job_id < 1:
            raise ConfigurationError("job_id must be a positive integer")
        if self.cluster is not None and (
            not isinstance(self.cluster, str) or TOKEN.fullmatch(self.cluster) is None
        ):
            raise ConfigurationError("cluster must be one safe site token")

    def __str__(self) -> str:
        return f"{self.job_id}" if self.cluster is None else f"{self.job_id};{self.cluster}"


@dataclass(frozen=True, slots=True)
class SchedulerEvidence:
    """Allocation-level evidence combined from squeue and anchored sacct history.

    ``retained`` means the scheduler still holds the work (queued, running, held, requeued) and
    blocks retry. ``problem`` is set when this allocation's evidence was contradictory; the
    state is then ``UNKNOWN`` and ``retained`` is true, which withholds only its Tasks.
    """

    state: AllocationState
    raw_state: str | None = None
    accounting_state: str | None = None
    exit_code: str | None = None
    reason: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    retained: bool = False
    problem: str | None = None


@dataclass(frozen=True, slots=True)
class StepEvidence:
    """Per-Task evidence from that Task's own ``srun`` step."""

    state: AllocationState
    raw_state: str
    exit_code: str | None = None


@dataclass(frozen=True, slots=True)
class Observation:
    """Everything observed for one accepted allocation; ``steps`` is indexed by slot."""

    allocation: SchedulerEvidence
    steps: tuple[StepEvidence | None, ...] = ()
