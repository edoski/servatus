# ADR 0002: Publish through a durable POSIX transaction

Status: accepted (revised for 0.12)

## One stage design

Every publication builds in one stage: a uniquely named, owner-only (0700) directory beside the
destination, created under the pinned destination parent. All filesystem work goes through pinned
descriptors (`_fs.Pin`): a descriptor plus the `stat` entry it was opened against, verified on
construction and closed on any verification failure, so no descriptor outlives a failed check.
Ownership checks compare against `os.geteuid()`.

- `publish(destination, build, *, retire=None, mode=None)` hands the builder a `Draft` whose path is
  the stage itself. The stage directory becomes the destination.
- `publish_file(destination, write, *, mode=None)` hands the writer `<stage>/<destination name>`.
  The writer may create that member any way it likes, including writing a temporary sibling and
  renaming it, or a library that insists on the real suffix (`np.save`, `savefig`). After the writer
  returns, Servatus pins the member, requires a single-link regular file, sets its mode, syncs it,
  and commits it out of the stage. The rest of the stage, including any writer leftovers, is then
  discarded.

Both operations reject an existing destination before entering the callback; the no-replace commit
remains authoritative against a later race. Creating a stage does not sync its parent; only
validated content is made durable before commit, and the destination parent is synced after it.

Invalid caller input (an unsafe or absolute draft path, a missing destination parent, a missing
link or `retire` source, a `retire` that is not a distinct sibling, a writer that creates nothing,
a `link_tree` source that contains the draft, a mode outside `0..0o777`, non-bytes Workspace
identity) raises `ConfigurationError`; a failed builder or writer discards its stage.

## Errors

Every public operation runs inside one error boundary (`_fs.Boundary`) where an `OSError` becomes
the error a caller can act on, so no raw `OSError` escapes publication, Workspaces, or the campaign
store. A missing entry is `ConfigurationError` (`NotFound` in the store); a denied, read-only,
overlong, or otherwise unusable location is `ConfigurationError`; a cross-device operation is
`CrossDeviceError`; anything else (no space, quota, I/O errors, descriptor exhaustion) is
transient and `Unavailable`. A file or directory its owner cannot read is `ConfigurationError`
("not readable by its owner"). `UnsafeFilesystem` is reserved for the explicit substitution, type,
and ownership checks, so it always means "investigate". Exceptions raised by application
callbacks (builders, writers) pass through the boundary with their identity intact.

A `Draft` is invalidated when the builder returns; later use raises `RuntimeError`, so a leaked
draft cannot mutate a committed destination. `Draft.link(source, destination)` hard-links one
regular file to a safe relative path. `Draft.link_tree(source_dir, destination=".")` walks a source
tree through pinned descriptors and hard-links every regular file, creating directories as needed;
symlinks and special files are rejected. Hard links alias their source inode: an in-place rewrite of
the source after publication changes the published content. Applications that keep writing a
source must replace it (new file plus rename) rather than rewrite it.

## Modes

Stages and every directory Servatus creates start owner-only (0700). Just before commit, every
directory of a `publish` tree is set to `mode` (default `0o777 & ~umask`, read without changing the
process umask); files inside keep the modes their builder gave them, and hard-linked files keep
their source's mode. A `publish_file` member is set to `mode` (default `0o666 & ~umask`),
overriding any mode the writer chose. Nothing is group- or world-visible before it is complete.

## Commit

Linux uses `renameat2(RENAME_NOREPLACE)`, looked up once: through the libc symbol when present,
otherwise through `syscall(SYS_renameat2)` on architectures whose syscall number is known (x86-64
and AArch64). When neither is available the call is treated as `ENOSYS`; it is never
`UnsupportedPlatform`. macOS uses `renameatx_np(RENAME_EXCL)`. Linux falls back only when the native
call reports `EINVAL`, `ENOSYS`, or `EOPNOTSUPP`/`ENOTSUP`, after verifying the exact source inode
relative to its pinned directory. Elsewhere those errors raise `UnsupportedPlatform`.

The regular-file fallback installs the member with a hard link, which remains kernel-exclusive, and
verifies the published inode; the stage, which still holds the member's original name, is then
discarded and the parent synced. A directory fallback requires an
owner-controlled parent and the exclusive directory lock (below) on it. Under that lock
Servatus rechecks parent ownership and permissions and the source inode, checks the destination is
absent, renames, and verifies the published inode. The lock is released before the parent sync, so
a slow remote sync cannot convoy independent destinations. Correctness of the fallback requires
every same-account publisher on every client to use Servatus, and the mount to provide one coherent
`flock` domain and stable inode identities. Cross-device and unexpected native errors never enter
the fallback.

## Durability

Files and directories are synced before commit and the parent after it. On macOS the sync helper
uses `fcntl(F_FULLFSYNC)`, because plain `fsync` there does not flush the drive cache; elsewhere it
uses `os.fsync`. A crash or sync failure after rename can leave a complete destination visible with
unconfirmed directory durability; callers treat that outcome as ambiguous and validate the
destination before retrying.

Work, stages, link sources, and destinations share one filesystem. Servatus never copies.

## Directory locks

Coordinating entries of one directory (Workspace parents, removal, and the directory-commit
fallback) takes an exclusive `flock` on a dedicated read-only handle to the pinned directory, so
closing the handle releases it. Some filesystems, notably NFS, refuse `flock` on a directory handle
(`EBADF`, `ENOLCK`, `EOPNOTSUPP`, `EINVAL`). There Servatus creates and locks an owner-only
(`0600`) regular file `.servatus.lock` inside the directory, and after locking requires the name to
still bind the locked inode. The file is never removed: unlinking a lock file would let a later
locker create a new inode and split the holders. Because the result of the directory `flock`
depends only on the filesystem, every actor on one filesystem takes the same path. The lock file
must be owned by the effective user with no group or world permissions, so a directory coordinated
this way cannot be shared between users. When neither lock works, the operation raises
`UnsupportedPlatform`.

## Workspaces, cleanup, and residue

A `Workspace` is identity-bound private work beside the destination, retained after failure and
removed only after a committed, parent-synced publication. Its container, work directory, lifecycle
lock, and identity file must belong to the effective user with no group or world permissions; the
identity record stores the exact container, lock, and work inode pins. `Workspace.discard()` takes
the exclusive lifecycle lease and removes the pinned private work without publishing.

Entering a Workspace whose destination already exists reclaims leftover private work under the
lifecycle lock (for example, after a crash between commit and cleanup) and then raises
`DestinationExists`. Only work bound to the same identity, or never initialized, is reclaimed; a
failed reclaim is noted on the error and the work is kept. Work bound to a different identity raises
`WorkspaceConflict`, whose message names the private path so the operator can inspect it.

A container is removed `work` first and `.identity` last, so an interrupted removal never leaves
work without the identity that owns it. A remnant whose `.identity` outlived its `work` is finished
by the next opener with the same identity, which then starts afresh; any other identity is refused.
A container whose `work` holds entries but which has no `.identity` is never adopted: entering
raises `WorkspaceConflict` naming the private path. An empty `work` without identity is the trace of
an initialization that stopped before exposing work, and is adopted.

Cleanup walks the pinned tree relative to descriptors without following links, rechecks each name
binding before removal, and removes the root only while it still names the expected inode. A
directory lacking owner read or search permission is first made owner-accessible (`u+rwx`) by
name relative to its pinned parent after checking the binding, and its inode is verified once
opened. A moved, substituted, or unremovable tree remains as residue. Directory publication may
also retire one existing owner-only destination sibling (`retire=`) only after the destination
commit is durable.
Every committed publication with unprovable cleanup returns `cleanup_pending=True` and makes one
best-effort `RuntimeWarning`; cleanup state never turns a commit into apparent failure.

Application callbacks own contents and validation and must finish mutating before returning.
Servatus verifies pathname and inode identity and syncs contents, but does not claim to exclude
concurrent content writers.
