"""Durable Slurm campaigns of opaque Tasks.

Start with ``Campaign.create``, ``Campaign.open``, or ``Campaign.ensure``. A ``Profile`` pairs a
``Target`` with per-Task ``Resources``; ``Campaign.plan`` returns a reviewable ``Plan`` and
``Campaign.submit`` executes exactly that plan. ``connect``, ``Transport``, and ``Completed`` are
the scheduler transport seam (see ``servatus.testing.FakeScheduler``); ``diagnose`` checks that a
profile's cluster has what Servatus needs, and ``to_document`` turns results into JSON-compatible
data.
"""

from ._campaign import Campaign, Connect, ResultProbe
from ._config import Apptainer, Profile, Resources, Target, Task
from ._doctor import Check, Diagnosis, Verdict, diagnose
from ._evidence import Execution, Observation, StepEnd, StopReason
from ._plan import Decision, Plan, PlannedAllocation, capacity
from ._policy import Hold, Retry
from ._remote import Completed, Transport, connect
from ._results import (
    LogSnapshot,
    Receipt,
    ReconcileResult,
    ShapeCheck,
    SubmitResult,
    UnattemptedAllocation,
    UnresolvedSubmission,
    WaitingAllocation,
    to_document,
)
from ._state import AcceptanceState, Link
from ._status import AttemptStatus, ResultState, RunTime, Status, TaskStatus

__all__ = [
    "AcceptanceState",
    "Apptainer",
    "AttemptStatus",
    "Campaign",
    "Check",
    "Completed",
    "Connect",
    "Decision",
    "Diagnosis",
    "Execution",
    "Hold",
    "Link",
    "LogSnapshot",
    "Observation",
    "Plan",
    "PlannedAllocation",
    "Profile",
    "Receipt",
    "ReconcileResult",
    "Resources",
    "ResultProbe",
    "ResultState",
    "Retry",
    "RunTime",
    "ShapeCheck",
    "Status",
    "StepEnd",
    "StopReason",
    "SubmitResult",
    "Target",
    "Task",
    "TaskStatus",
    "Transport",
    "UnattemptedAllocation",
    "UnresolvedSubmission",
    "Verdict",
    "WaitingAllocation",
    "capacity",
    "connect",
    "diagnose",
    "to_document",
]
