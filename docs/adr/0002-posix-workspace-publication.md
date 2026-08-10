# ADR 0002: Publish through a durable POSIX transaction

Status: accepted

Servatus builds a directory in a unique destination-adjacent stage, rejects unsafe entries, and
recursively syncs its contents before committing it as the destination. `publish_file` uses the same
transaction for one regular file: it creates an empty stage with ordinary umask-controlled
permissions, pins its descriptor and inode, requires in-place writing, and syncs it before commit.
Both operations then sync the destination parent.

Linux first uses `renameat2(RENAME_NOREPLACE)` and macOS uses descriptor-relative
`renameatx_np(RENAME_EXCL)`. Linux falls back only when the native call reports `EINVAL`, `ENOSYS`,
or `EOPNOTSUPP`, after re-verifying the pinned parent and exact source inode. A regular file is
installed with a same-parent hard link, which remains kernel-exclusive, then the exact stage is
removed; both parent-directory changes are synced. If stage removal reports failure after the source
name disappeared, Servatus retries the parent sync before reporting cleanup pending. It never removes
a source found under a substituted inode.

A directory fallback requires an owner-controlled parent and an exclusive advisory `flock` on its
pinned descriptor. Under that lock Servatus re-verifies the parent and source, checks the destination
is absent, performs a descriptor-relative rename, verifies the published inode, and syncs the parent.
The lock lives on a dedicated handle and is released by closing that handle, so an unlock error cannot
mask a verified, parent-synced commit. Correctness requires every same-account publisher on every
client to use Servatus and the filesystem mount to provide one coherent `flock` domain and stable
inode identities across those clients; local-only or disabled lock modes and unstable cross-client
inodes are unsupported. Same-account code that ignores the lock is outside the contract. Locking or
verification failure closes the transaction. Cross-device and unexpected native errors do not enter
the fallback.

Work, stages, link sources, and destinations must share one filesystem. Servatus never copies during
publication.

Application callbacks own contents and validation. File writers may change the mode but may not
unlink, replace, or change the type of the stage they receive.

An identity-bound workspace is retained after build failure and removed only after a committed,
parent-synced publication. If fallback identity installation commits but stage removal or its
durability cannot be proved, the installed identity remains valid and Servatus warns that private
cleanup remains pending. Cleanup state is reported separately from publication success.
