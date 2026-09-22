from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._model import SubmitResult


class ServatusError(Exception):
    """Base class for Servatus contract failures."""


class PublicationError(ServatusError):
    """A draft could not be safely published."""


class DestinationExists(PublicationError):
    """The requested destination or draft entry already exists."""


class CrossDevicePublication(PublicationError):
    """Publication would cross a filesystem boundary."""


class UnsafePublication(PublicationError):
    """A path or filesystem entry violates the publication contract."""


class UnsupportedPlatform(PublicationError):
    """The platform cannot provide an atomic no-replace commit."""


class WorkConflict(ServatusError):
    """Existing private work is bound to another identity."""


class WorkspaceBusy(ServatusError):
    """Another writer currently owns the workspace."""


class CampaignError(ServatusError):
    """A campaign operation could not preserve its lifecycle contract."""


class ConfigurationError(CampaignError):
    """A Task, Profile, target, or resource request is invalid."""


class TaskConflict(CampaignError):
    """Campaign state or roster authoring violates its contract."""


class PlanError(CampaignError):
    """A submission plan is infeasible, stale, foreign, or changed."""


class SubmissionError(CampaignError):
    """Submission stopped; result preserves any partial or uncertain acceptance."""

    def __init__(self, message: str, *, result: SubmitResult | None = None) -> None:
        super().__init__(message)
        self.result = result


class ReconciliationError(CampaignError):
    """A bounded scheduler query could not prove one allocation identity."""


class ObservationError(CampaignError):
    """A bounded Campaign observation was unavailable or untrustworthy."""
