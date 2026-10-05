"""Atomic, durable, no-replace publication of files and directories on one POSIX filesystem.

A destination is either absent or one complete regular file or directory; it is never
overwritten. Work, hard-link sources, private stages, and the destination must share one
filesystem (Linux or macOS). See `publish`, `publish_file`, and `Workspace`.

Deliberate changes from 0.11 (all other ADR 0002/0004 guarantees are unchanged):

- `publish_file` writers receive `<private 0700 stage>/<destination name>`, which does not exist
  yet, instead of an existing empty file to write in place. Any strategy works (suffix-appending
  savers, temp file then rename); other stage leftovers are discarded. The result must be a
  single-link regular file.
- Modes are applied just before commit: every published directory gets `mode` (default
  `0o777` minus the umask) and a published file gets `mode` (default `0o666` minus the umask),
  overriding any mode a writer chose. Stages and Servatus-created directories start at 0700.
- A `Draft` is unusable once its builder returns.
- On Linux a libc without `renameat2` uses the raw syscall where the number is known, otherwise
  the documented fallback; it is no longer `UnsupportedPlatform`.
- Entering a `Workspace` whose destination already exists first removes leftover private work
  bound to the same identity (or never initialized), then raises `DestinationExists`.
- A failed identity-stage cleanup sync is reported once, without a retry.
- Workspace containers are removed work first and `.identity` last. The same identity finishes an
  interrupted removal on its next entry; a container holding work without an identity is never
  adopted (`WorkspaceConflict` naming the private path).
- No raw `OSError` escapes: a missing, unwritable, or overlong location is `ConfigurationError`;
  transient failures (no space, quota, I/O errors, descriptor exhaustion) are `Unavailable`;
  `UnsafeFilesystem` means substitution, type, or ownership. Exceptions raised by builders and
  writers propagate unchanged. A file unreadable by its owner is `ConfigurationError`.
- `Draft.link_tree` rejects a source tree containing the draft (`ConfigurationError`).
- Where `flock` on a directory handle fails (NFS: EBADF, ENOLCK, EOPNOTSUPP, EINVAL), parent
  coordination locks an owner-only `.servatus.lock` file in that directory instead; it is left in
  place, and must not be shared between users.
"""

from ._transaction import Draft, Publication, publish, publish_file
from ._workspace import Workspace

__all__ = ["Draft", "Publication", "Workspace", "publish", "publish_file"]
