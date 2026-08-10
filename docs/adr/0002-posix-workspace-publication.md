# ADR 0002: Publish through a durable POSIX transaction

Status: accepted

Servatus builds in a unique destination-adjacent stage, rejects unsafe entries, recursively syncs
content, and atomically renames the directory into an absent destination with the platform's
no-replace primitive. It then syncs the destination parent.

Linux uses `renameat2(RENAME_NOREPLACE)` and macOS uses descriptor-relative
`renameatx_np(RENAME_EXCL)`. Unsupported systems and filesystems fail closed. Work, stages, link
sources, and destinations must share one filesystem; Servatus never weakens the contract with a
copy or check-then-rename fallback.

An identity-bound workspace is retained after build failure and removed only after a committed,
parent-synced publication. Cleanup failure is reported separately from publication success.
