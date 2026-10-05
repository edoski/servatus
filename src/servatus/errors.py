"""Servatus errors, grouped by what the caller can do about them."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .campaign._results import SubmitResult


class ServatusError(Exception):
    """Base class for every Servatus failure."""


class ConfigurationError(ServatusError):
    """An input is invalid. Fix the Task, Profile, path, or argument and try again."""


class CrossDeviceError(ConfigurationError):
    """Work, sources, and destination must share one filesystem."""


class UnsupportedPlatform(ConfigurationError):
    """The platform or filesystem cannot provide the required atomic operation."""


class NotFound(ServatusError):
    """A campaign, allocation, Task, or file does not exist."""


class Conflict(ServatusError):
    """The request contradicts current state. Inspect it, then change the request."""


class StalePlan(Conflict):
    """The Campaign changed after the plan was made. Plan again."""


class DestinationExists(Conflict):
    """The publication destination already exists."""


class WorkspaceConflict(Conflict):
    """Existing private work is bound to a different identity."""


class PlanRefused(ServatusError):
    """The eligibility policy needs an explicit operator decision for these Task keys."""

    def __init__(self, message: str, *, keys: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.keys = keys


class Busy(ServatusError):
    """Another actor holds or changed the resource. Retry soon."""


class Unavailable(ServatusError):
    """The cluster or a remote command failed or timed out. Retry later."""


class IntegrityError(ServatusError):
    """Durable state, the filesystem, or scheduler evidence is untrustworthy. Investigate."""


class CorruptState(IntegrityError):
    """Durable Campaign state is malformed or violates its invariants."""


class UnsafeFilesystem(IntegrityError):
    """A filesystem entry was substituted, moved, or has unsafe ownership or permissions."""


class EvidenceConflict(IntegrityError):
    """Scheduler output is malformed, unrelated, or contradictory."""


class ReconciliationError(ServatusError):
    """Scheduler evidence could not prove one allocation identity. Resolve it explicitly."""


class SubmissionInterrupted(ServatusError):
    """Submission stopped early. ``result`` records receipts and uncertain allocations."""

    def __init__(self, message: str, *, result: SubmitResult) -> None:
        super().__init__(message)
        self.result = result
