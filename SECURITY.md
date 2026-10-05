# Security policy

Please report vulnerabilities privately through GitHub's security-advisory form for this
repository. Do not include sensitive paths, credentials, or research data in a public issue.

Only the latest released version receives security fixes.

## Trust model

Servatus is an unprivileged user library, not an authorization, sandboxing, tenant-isolation, or
cluster-policy layer. Its publication transaction protects against accidental partial visibility,
overwrites, and ordinary lifecycle races. Callers remain responsible for destination-parent
permissions, trusted builders and workers, application validation, filesystem guarantees, and
scheduler policy. Arbitrary code running as the same Unix account is outside the threat model.

## Publication

Stages are private owner-only directories. Published directories, and a `publish_file` file,
receive their final `mode` only just before commit (a file's mode overrides whatever its writer
set), so nothing is group- or world-visible while incomplete. Files inside a published directory
keep the modes they were written or hard-linked with. Ownership checks compare against the
effective user id.

Builders and file writers must finish mutating content before returning; the `Draft` is invalidated
then. `Draft.link()` and `Draft.link_tree()` select source inodes at the hard-link operation, so a
safe replacement of a source path before that operation may be selected. Servatus checks each linked
entry without following symlinks, verifies pathname/inode identity, and syncs content. It does not
detect or exclude concurrent content writers. A hard-linked result shares its source inode:
rewriting the source in place after publication changes the published content. The owner-only draft
namespace must remain quiescent during each link call.

Optional directory retirement (`retire=`) accepts one existing owner-only destination sibling.
Servatus pins it before the builder, commits and syncs the destination first, then removes only the
pinned tree and syncs the parent again. Missing, moved, substituted, newly permissive, or
unremovable trees after commit are preserved as cleanup residue and cannot turn a committed
publication into apparent failure.

The hidden Workspace container is the lifecycle trust root. Its container, work directory,
lifecycle lock, and identity file must belong to the effective user with no group or world
permissions and are reverified through pinned descriptors. Initialization syncs the private
hierarchy and its parent before identity becomes authoritative; no sync runs under shared parent
coordination. Cleanup, `discard()`, and residue reclaim walk only the pinned tree without following
links, check name bindings before descriptor-relative removal, and preserve a moved or substituted
root as residue. These checks do not make inspection and unlink atomic against a concurrent
same-account namespace writer. Keep destination parents private.

On Linux, when `renameat2(RENAME_NOREPLACE)` is unavailable (no libc wrapper and no known raw
syscall number) or the filesystem rejects it (`EINVAL`, `ENOSYS`, `EOPNOTSUPP`/`ENOTSUP`),
regular-file publication keeps kernel-enforced create-if-absent semantics through a hard link.
Directory publication instead uses a check-and-rename transaction under an exclusive advisory lock
on the pinned parent, released before the parent sync. A failure or crash after rename can leave a
complete visible destination with unconfirmed directory durability; treat that outcome as ambiguous
and validate the destination before retrying. The fallback requires a parent owned by the effective
user and not group- or world-writable, every same-account publisher on every client to use
Servatus, and a mount with coherent `flock` and stable inode identities.

## Campaigns

Task arguments, stdin, and environment values are written into the submitted batch script (stdin as
`printf` literals, environment as `APPTAINERENV_*` assignments or `env -i` arguments). Cluster
administrators and accounting systems may read them. Do not put credentials or other secrets in
Tasks. The batch script creates no scratch files.

Values in a repository-owned `SERVATUS.toml` are editable user-side guardrails, not an enforcement
boundary; Profile labels are nonbinding provenance. Slurm remains authoritative for identity,
admission, isolation, allocation, accounting, and billing. GPU steps forward Slurm's step-local
device visibility with PCI bus ordering; this does not replace site GPU isolation.

Scheduler commands run under a scrubbed environment with a fixed path, C locale, and UTC timezone;
local scheduler or timezone overrides are not forwarded. Over SSH, Servatus uses batch mode (never
prompting), and discards everything printed before its per-call marker, so login banners and
shell-startup output cannot be mistaken for scheduler output. SSH authentication, host keys, and
connection sharing are configured in the user's OpenSSH configuration.

Campaign state is an owner-only directory. It keeps complete Task arguments, stdin, and environment
for retry; keep it private.

Plan files (`servatus.plan/1`) retain the resolved Target and Resources, Task keys with their hold
reasons, retry choices, duplicate-risk acknowledgements, and the nonce from which allocation
identities derive. They exclude Task arguments, stdin, and environment
but remain private operational data. `Plan.save` and `servatus plan --output` write them with mode
`0600` and never overwrite an existing file. `--show-scripts` prints complete scripts, including
Task payloads; treat that output as sensitive. Submission rebuilds each plan from its decision and
the current state and refuses a plan that does not match.

`Status.to_json()` (`servatus status --json`) is a diagnostic snapshot, not a redacted record. It
includes Task keys, result states, allocation identities, retry choices, Profile labels, Slurm job
ids and clusters, raw scheduler and accounting states, exit codes, scheduler reasons, timestamps,
and any evidence problem text. It excludes Task arguments, stdin, environment values, Target values,
scripts, log content, and application output. Scheduler text, keys, and labels may themselves
identify private work; keep snapshots private. Snapshots carry no execution authority.

Submission checks the scheduler connection and the plan's eligibility before recording anything,
then syncs unresolved intent before each `sbatch`. A failure after launch does not prove rejection.
`SubmissionInterrupted.result` distinguishes confirmed receipts, unresolved work, unattempted
allocations, and a job observed but not durably recorded. Reconcile uncertain acceptance before
retrying; unknown accepted work is retried only after an explicit duplicate-risk acknowledgement.

## Logs

Log snapshots are sensitive, untrusted binary data. They may contain credentials, research data,
terminal control sequences, or other hostile output. Use `servatus logs --output FILE`, which writes
an owner-only (`0600`) file; the command refuses to write log bytes to a terminal. Inspect logs with
a safe viewer.

Servatus accepts no caller-supplied remote path or command for logs. It derives one path inside the
Attempt's `log_root` from a validated accepted Attempt, but that namespace remains controlled by the
same cluster account; Servatus does not pin a remote inode or authenticate log content. Remote log
failures surface as `Unavailable` without log bytes, remote paths, or remote stderr in the message.
Log bytes are never stored in Campaign state, plans, or status snapshots.
