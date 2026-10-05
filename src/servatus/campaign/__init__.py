"""Durable Slurm campaigns of opaque Tasks.

Start with ``Campaign.create``, ``Campaign.open``, or ``Campaign.ensure``. A ``Profile`` pairs a
``Target`` with per-Task ``Resources``; ``Campaign.plan`` returns a reviewable ``Plan`` and
``Campaign.submit`` executes exactly that plan. ``connect``, ``Transport``, and ``Completed`` are
the scheduler transport seam (see ``servatus.testing.FakeScheduler``); ``ping`` checks a Target's
scheduler, and ``to_document`` turns results into JSON-compatible data.
"""

from ._campaign import Campaign, Connect, ResultProbe, ping
from ._config import Apptainer, Profile, Resources, Target, Task
from ._evidence import AllocationState, JobRef, SchedulerEvidence, StepEvidence
from ._plan import Decision, Plan, PlannedAllocation, capacity
from ._policy import Hold, Retry
from ._remote import Completed, Transport, connect
from ._results import (
    LogSnapshot,
    Receipt,
    ShapeCheck,
    SubmitResult,
    UnattemptedAllocation,
    UnresolvedSubmission,
    to_document,
)
from ._state import AcceptanceState
from ._status import AttemptStatus, ResultState, Status, TaskStatus

__all__ = [
    "AcceptanceState",
    "AllocationState",
    "Apptainer",
    "AttemptStatus",
    "Campaign",
    "Completed",
    "Connect",
    "Decision",
    "Hold",
    "JobRef",
    "LogSnapshot",
    "Plan",
    "PlannedAllocation",
    "Profile",
    "Receipt",
    "Resources",
    "ResultProbe",
    "ResultState",
    "Retry",
    "SchedulerEvidence",
    "ShapeCheck",
    "Status",
    "StepEvidence",
    "SubmitResult",
    "Target",
    "Task",
    "TaskStatus",
    "Transport",
    "UnattemptedAllocation",
    "UnresolvedSubmission",
    "capacity",
    "connect",
    "ping",
    "to_document",
]
