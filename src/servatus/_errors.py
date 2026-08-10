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
