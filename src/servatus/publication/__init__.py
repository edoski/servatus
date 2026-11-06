"""Atomic, durable, no-replace publication of files and directories on one POSIX filesystem.

A destination is either absent or one complete regular file or directory; it is never
overwritten. Work, hard-link sources, private stages, and the destination must share one
filesystem (Linux or macOS). See `publish`, `publish_file`, and `Workspace`.
"""

from ._transaction import Draft, Publication, publish, publish_file
from ._workspace import Workspace

__all__ = ["Draft", "Publication", "Workspace", "publish", "publish_file"]
