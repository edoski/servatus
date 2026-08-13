# ADR 0002: Publish through a durable POSIX transaction

Status: accepted

Servatus builds a directory in a unique destination-adjacent stage, rejects unsafe entries, and
recursively syncs its contents before committing it as the destination. `publish_file` uses the same
transaction for one regular file: it creates an empty stage with ordinary umask-controlled
permissions, pins its descriptor and inode, requires in-place writing, and syncs it before commit.
Creating either disposable stage does not sync its parent; only validated content is made durable
before commit. Both operations inspect the pinned parent and reject an existing destination before
entering the application callback, then sync the destination parent after commit. The no-replace
commit remains authoritative against a later race.

Linux first uses `renameat2(RENAME_NOREPLACE)` and macOS uses descriptor-relative
`renameatx_np(RENAME_EXCL)`. Linux falls back only when the native call reports `EINVAL`, `ENOSYS`,
or `EOPNOTSUPP`, after re-verifying the pinned parent and exact source inode. A regular file is
installed with a same-parent hard link, which remains kernel-exclusive, then the exact stage is
removed; both parent-directory changes are synced. If stage removal reports failure after the source
name disappeared, Servatus retries the parent sync before reporting cleanup pending. It never removes
a source found under a substituted inode.

A directory fallback requires an owner-controlled parent and an exclusive advisory `flock` on its
pinned descriptor. Under that lock Servatus re-verifies the parent and source, checks the destination
is absent, performs a descriptor-relative rename, and verifies the published inode. Servatus then
closes the dedicated lock handle before syncing the parent, so an unbounded remote directory sync
cannot retain the distributed lock or convoy independent destinations. Publication returns only
after the parent sync. A crash or sync failure after rename can leave the complete destination
visible with unconfirmed directory durability, so the caller must treat the failed operation as
ambiguous and validate the canonical destination before retrying. Correctness requires every
same-account publisher on every client to use Servatus and the filesystem mount to provide one
coherent `flock` domain and stable inode identities across those clients; local-only or disabled lock
modes and unstable cross-client inodes are unsupported. Same-account code that ignores the lock is
outside the contract. Locking or verification failure closes the transaction. Cross-device and
unexpected native errors do not enter the fallback.

Work, stages, link sources, and destinations must share one filesystem. Servatus never copies during
publication.

`Draft.link()` selects its source inode at the atomic hard-link operation. It then inspects the
linked draft entry without following symlinks and accepts only a same-filesystem regular file. A safe
source-path replacement before the link operation may therefore be selected; there is no preliminary
source-inode anchor. The owner-only draft namespace is trusted and must remain quiescent during the
call; hostile same-account replacement of the destination leaf during `Draft.link()` is outside the
contract.

Application callbacks own contents and validation and must finish content mutations before
returning. Servatus verifies pathname/inode identity and syncs contents afterward, but does not use
size or modification-time comparisons to claim concurrent-writer exclusion. File writers may change
the mode but may not unlink, replace, or change the type of the stage they receive.

Directory publication may also pin one existing, distinct, owner-only destination sibling before
the builder. Every precommit failure preserves this retained tree. After the destination commit and
parent sync, Servatus removes only the exact pinned tree through descriptor-relative traversal and
syncs the parent again. A missing, moved, substituted, newly permissive, or unremovable tree remains
as cleanup residue. The caller must make the tree quiescent before publication; retirement is not a
writer lease, callback, token, registry, or application finalizer.

An identity-bound workspace is retained after build failure and removed only after a committed,
parent-synced publication. Its container, work directory, lifecycle lock, and identity file must be
owned by the effective user and expose no group or world permissions. Its private identity record
stores the exact container, lock, and work inode pins. Live opens enforce local type, device, and
pathname identity. Before first identity commit, Servatus syncs the lock and work entries, their
container, and the container's parent; a durable identity is not synced again merely on reopen.
Cleanup opens the expected root, walks and removes entries relative to pinned directory descriptors
without following links, reverifies each name binding, and removes the root only while it still
names the expected inode. A moved, substituted, or unremovable tree remains as cleanup residue. If
fallback identity installation commits but stage removal or its durability cannot be proved, the
installed identity remains valid and Servatus warns that private cleanup remains pending. Every
public committed publication with stage, Workspace, or retained-tree residue returns
`cleanup_pending=True` and makes one best-effort `RuntimeWarning`; warning filters and hooks cannot
turn that commit into apparent failure. Cleanup state is authoritative and separate from
publication success.
