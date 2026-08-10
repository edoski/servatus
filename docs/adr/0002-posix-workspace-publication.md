# ADR 0002: Publish through a durable POSIX transaction

Status: accepted

Servatus builds a directory in a unique destination-adjacent stage, rejects unsafe entries,
recursively syncs its contents, and atomically renames it into an absent destination with the
platform's no-replace primitive. `publish_file` uses the same transaction for one regular file: it
creates an empty stage with ordinary umask-controlled permissions, pins its descriptor and inode,
requires in-place writing, and syncs it before commit. Both operations then sync the destination
parent.

Linux uses `renameat2(RENAME_NOREPLACE)` and macOS uses descriptor-relative
`renameatx_np(RENAME_EXCL)`. Unsupported systems and filesystems fail closed. Work, stages, link
sources, and destinations must share one filesystem; Servatus never weakens the contract with a
copy or check-then-rename fallback.

Application callbacks own contents and validation. File writers may change the mode but may not
unlink, replace, or change the type of the stage they receive.

An identity-bound workspace is retained after build failure and removed only after a committed,
parent-synced publication. Cleanup failure is reported separately from publication success.
