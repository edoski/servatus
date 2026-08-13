# Security policy

Please report vulnerabilities privately through GitHub's security-advisory form for this
repository. Do not include sensitive paths, credentials, or research data in a public issue.

Only the latest released version receives security fixes.

Servatus is an unprivileged user library, not an authorization, sandboxing, tenant-isolation, or
cluster-policy layer. Its publication transaction protects against accidental partial visibility,
overwrites, and ordinary lifecycle races. Callers remain responsible for destination-parent
permissions, trusted builders, application validation, filesystem guarantees, and scheduler policy.

Builders and file writers must finish mutating draft contents before returning. `Draft.link()`
selects its source inode at the atomic hard-link operation, so a safe replacement of the source path
before that operation may be selected. Servatus then checks the linked entry without following
symlinks, verifies pathname/inode identity, and syncs content. It does not detect or exclude
concurrent content writers. The owner-only draft namespace must remain quiescent during each
`Draft.link()`; hostile same-account replacement of its destination leaf during that call and
content mutation after the callback returns are outside the contract.

Optional directory retirement accepts one existing owner-only destination sibling. Servatus pins
the parent and source before the builder, commits and syncs the destination first, then removes only
the pinned source tree and syncs the parent again. The caller must stop all source writers before the
transaction. Missing, moved, substituted, newly permissive, or unremovable sources after commit are
preserved as cleanup residue and cannot turn committed publication into apparent failure.

The hidden Workspace container is its lifecycle trust root. Servatus requires the container, its
work directory, lifecycle lock, and identity file to belong to the effective user with no group or
world permissions, and reverifies them through pinned descriptors. Initialization syncs the private
hierarchy before identity becomes authoritative. Cleanup walks only the pinned tree without
following links, reverifies every name binding, and preserves a moved or substituted root as cleanup
residue. Arbitrary code running as the same Unix account can rename and recreate the complete trust
root and is outside this unprivileged library's threat model. Keep destination parents private and
treat workers and builders as trusted code.

On Linux filesystems without `renameat2(RENAME_NOREPLACE)`, regular-file publication retains
kernel-enforced create-if-absent semantics through a same-directory hard link. Directory publication
instead uses a check-and-rename transaction under an exclusive advisory lock on the pinned parent
descriptor. Servatus requires that parent to be owned by the effective user and not group- or
world-writable. This contract requires every same-account publisher on every client to use Servatus
and the filesystem mount to provide coherent `flock` and stable inode identities across all those
clients. The identity record stores only exact container, lock, and work inode pins; live opens still
enforce local type, device, and pathname-to-inode identity. Local-only or disabled locking and
unstable cross-client inodes are unsupported. A same-account process that ignores the lock remains
outside the threat model.

Campaign task arguments and stdin are embedded in the submitted batch script. Redaction from
ordinary local summaries does not make them secret; do not submit credentials or other secrets.
Resolved values from repository-owned `SERVATUS.toml` are editable user-side guardrails, not an
enforcement boundary. Profile labels are nonbinding provenance. Slurm remains
authoritative for identity, admission, isolation, allocation, accounting, and billing. Servatus
clears the remote command environment and supplies only a fixed path, C locale, and UTC timezone;
local scheduler or timezone overrides are not forwarded.

Canonical plan files retain exact target/resource values, scheduler observations, job identities,
and Task keys. They exclude Task arguments and stdin but are still private operational data. Store
them owner-only; the CLI publishes with mode `0600` and refuses overwrite. Unknown accepted work is
replayed only after a separate explicit duplicate-execution-risk acknowledgement.

Campaign log snapshots are sensitive, untrusted binary data. They may contain credentials,
research data, terminal control sequences, or other hostile output. Never render them directly in
a terminal; redirect CLI output to a private file or inspect it with a safe binary viewer. Servatus
accepts no caller-supplied remote path or command and derives one allocation-bound path from a
validated accepted Attempt, but the remote `log_root` namespace remains controlled by the same
cluster account. Servatus does not pin a remote inode or authenticate log content after Slurm writes
the path. A same-account process that replaces or mutates that path is outside this unprivileged
library's threat model.

Remote log failures are deliberately collapsed to a redacted `ObservationError`: returned error
messages and visible exception chains contain no log bytes, remote path, host, command, or remote
stderr. Log bytes are never stored in Campaign state, views, plans, or automatic structured output.
