"""Servatus: resumable Slurm campaigns and atomic publication of validated outputs."""

from importlib.metadata import version

from .campaign import Apptainer, Campaign, Profile, Resources, Retry, Target, Task
from .errors import ServatusError
from .publication import Draft, Publication, Workspace, publish, publish_file

__version__ = version("servatus")

__all__ = [
    "Apptainer",
    "Campaign",
    "Draft",
    "Profile",
    "Publication",
    "Resources",
    "Retry",
    "ServatusError",
    "Target",
    "Task",
    "Workspace",
    "__version__",
    "publish",
    "publish_file",
]
