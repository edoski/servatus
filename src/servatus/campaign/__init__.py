"""Durable Slurm campaigns of opaque Tasks.

Start with ``Campaign.create``, ``Campaign.open``, or ``Campaign.ensure``. A ``Profile`` pairs a
``Target`` with per-Task ``Resources``; ``Campaign.plan`` returns a reviewable ``Plan`` and
``Campaign.submit`` executes exactly that plan; a ``Continuation`` makes a plan continue its Tasks
across a chain of allocations under a lifetime run-time budget. ``connect``, ``Transport``, and
``Completed`` are the scheduler transport seam (see ``servatus.testing.FakeScheduler``);
``diagnose`` checks that a profile's cluster has what Servatus needs, and ``to_document`` turns
results into JSON-compatible data.
"""

from ._campaign import Campaign, Connect, ResultProbe
from ._config import (
    MAX_LINKS,
    YIELD_EXIT,
    Apptainer,
    Continuation,
    Profile,
    Resources,
    Target,
    Task,
)
from ._doctor import Check, Diagnosis, Verdict, diagnose
from ._evidence import Execution, Observation, StepEnd, StopReason
from ._plan import Decision, Plan, PlannedAllocation, Totals, capacity
from ._policy import Extension, Hold, Retry
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
    "MAX_LINKS",
    "YIELD_EXIT",
    "AcceptanceState",
    "Apptainer",
    "AttemptStatus",
    "Campaign",
    "Check",
    "Completed",
    "Connect",
    "Continuation",
    "Decision",
    "Diagnosis",
    "Execution",
    "Extension",
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
    "Totals",
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
