# Servatus

Run resumable work through Slurm and atomically publish validated outputs.

Servatus 0.7.0 combines durable publication with the native Slurm Campaign interface below.

```sh
pip install servatus
```

## Campaigns

A Campaign owns an ordered roster of opaque tasks. While open, reopening with the exact sequence is
idempotent and reopening with that exact prefix plus a nonempty suffix durably registers the new
tasks. Sealing ends authoring irreversibly. Removing, reordering, or changing any registered task
fails. Planning is local and deterministic. Submission records durable intent before contacting
Slurm, records the acceptance afterward, and stops on an ambiguous missing receipt rather than
risking duplicate work.

```python
from pathlib import Path, PurePosixPath

from servatus import Campaign, Profile, Task

campaign = Campaign.open(
    Path("state/training"),
    [Task("candidate-0", ("train", "--candidate", "0"), b'{"seed": 7}\n')],
)
campaign.seal()  # Seal fixed rosters; appendable campaigns may execute while still open.
profile = Profile.load(Path("SERVATUS.toml"), name="research")
view = campaign.inspect()  # Scheduler-only unless an application result probe is supplied.
plan = campaign.plan(profile, view=view)
# Review plan.allocations and plan.digest, then submit explicitly:
# validation = campaign.validate(plan)  # Optional, time-specific Slurm --test-only check.
receipts = campaign.submit(plan)
# Later processes reopen the durable roster without resupplying Tasks:
campaign = Campaign.load(Path("state/training"))
```

`ResourceRequest` has no defaults. CPU and MiB memory are positive, GPUs are a nonnegative whole
count, and time uses canonical `[days-]hours:minutes:seconds`. One request applies to every Task in
one Campaign. CPU-only, one-GPU, and one-process whole-multi-GPU tasks are supported. A project uses
separate Campaigns for different resource shapes.

The request retains the authored wall time for provenance. Plans and `sbatch` show Slurm's effective
limit, rounded upward once to whole minutes. The same rounding is applied before comparing a request
with the target ceiling; packed task count never multiplies wall time.

`SERVATUS.toml` is one repository-owned document containing one or more complete named execution
profiles. Its root contains exactly `profiles` and optional `default_profile`. Every profile is
validated even when it is not selected, and an explicit name overrides the default.

```toml
default_profile = "research"

[profiles.research.target]
host = "login.example.edu"
slurm_bin = "/opt/slurm/bin"
apptainer = "/usr/bin/apptainer"
image = "/cluster/images/project.sif"
work_root = "/cluster/work/project"
log_root = "/cluster/logs/project"
partitions = ["gpu"]
account = "research" # optional; qos and constraint are also optional
gpu_gres = "gpu"     # omit for a CPU-only target
max_tasks_per_allocation = 4
max_cpus_per_allocation = 128
max_memory_mib_per_allocation = 262144
max_gpus_per_allocation = 4 # use 0 when gpu_gres is omitted
max_time_limit = "7-00:00:00"
max_allocations_per_submit = 64
max_script_bytes = 4194304

[profiles.research.resources]
cpus_per_task = 32
memory_mib_per_task = 65536
gpus_per_task = 1
time_limit = "3-00:00:00"
```

Unknown keys, counted GRES, relative remote paths, unsafe site tokens, controls, booleans used as
integers, unlimited/zero resources, incomplete profiles, and conflicting GPU settings are rejected.
Profiles do not inherit, merge, search parent directories, consult environment variables, or use a
global store. The label is nonbinding provenance; exact resolved target and resource values own
Campaign compatibility. A target is a user-side mistake guard, not cluster authorization. Every
listed partition must fit one truthful conservative envelope.

The planner preserves authored order and uses the fewest balanced groups allowed by every declared
ceiling. An allocation containing `n` Tasks requests exactly `n*C` CPUs, `n*M` MiB, and `n*G` GPUs;
time remains `T`. A caller may lower packing with `tasks_per_allocation`, but a cap above feasible
capacity is rejected rather than clamped. Servatus never rounds up to node capacity.

`Campaign.tasks` returns the immutable authored Task tuple. Append-only growth is allowed only while
the roster is open and preserves target/resource lineage plus every accepted, ambiguous,
not-submitted, and retry attempt. Append and seal each increment Campaign revision, so older plans
become stale. Execution remains valid while open. `Campaign.seal()` is atomic and idempotent; a
sealed Campaign accepts only an exact-roster reopen.

Each accepted or explicitly not-submitted outcome records its mutation revision. This preserves
actual chronology when an older ambiguous Attempt is resolved after newer disjoint work.

Planning consumes one exact revision-bound `CampaignView`. Valid results are excluded.
Never-accepted missing or unobserved Tasks are selected. Accepted active work and ambiguous
acceptance are withheld. Terminal accepted work requires explicit retry; unknown accepted work also
requires explicit duplicate-execution-risk acknowledgement. Any older active or unknown accepted
attempt still governs retry safety.

The current development line uses Campaign schema 4 and plan schema 4. Campaign schema 3 is rejected
without migration; create a new Campaign when upgrading. Loading a plan regenerates it from its
typed frozen view without probing or scheduler contact and requires identical canonical bytes.

Each allocation runs one concurrent
`srun --exclusive --exact --nodes=1 --ntasks=1` step per Task. Each step receives its exact CPU,
MiB, and whole-GPU request and starts the target's immutable Apptainer image from `work_root`.
CPU-only work emits no GRES or `--nv`. Servatus never emits job-level exclusivity, overlap, all
memory, manual CUDA indices, ranks, or raw scheduler flags.

Slurm writes combined allocation stdout/stderr to
`log_root/<allocation_id>-%j.out` and each combined task stream to
`log_root/<allocation_id>-%j-<zero-based-slot>.out`. Slurm expands `%j` after assigning the job ID;
the immutable allocation identity prevents reused job numbers from aliasing attempts.

Actual simultaneous placement depends on truthful resource requests and site CPU/GRES topology.
Servatus does not silently inflate CPU requests, disable binding, or expose raw scheduler flags.

Task arguments and byte-exact stdin are embedded in the complete batch script before `sbatch`
acceptance. They are excluded from ordinary plan and inspection output, but are not secrets: cluster
administrators and accounting systems may be able to inspect them. Scheduler names expose only a
random Servatus allocation identity.

### CLI

The Python interface is authoritative. The CLI task JSONL adapter has exactly `key`, string-array
`args`, and `stdin_file` per line:

```sh
servatus plan TASKS.jsonl --campaign STATE_DIR --output PLAN.json --tasks-per-allocation 4
servatus plan TASKS.jsonl --campaign STATE_DIR --output RETRY.json --retry task-0
# Unknown accepted work requires the separate duplicate-risk acknowledgement:
servatus plan TASKS.jsonl --campaign STATE_DIR --output RISKY.json \
  --retry task-0 --allow-duplicate-risk task-0
# Explicit sensitive diagnostic; prints complete scripts, arguments, and payloads:
servatus plan TASKS.jsonl --campaign STATE_DIR --output PLAN.json --show-scripts
servatus seal STATE_DIR
servatus validate STATE_DIR PLAN.json
servatus submit STATE_DIR PLAN.json
servatus inspect STATE_DIR
servatus log STATE_DIR ALLOCATION_ID --task TASK_KEY --bytes 65536 > task.log
servatus reconcile STATE_DIR ALLOCATION_ID
servatus resolve STATE_DIR ALLOCATION_ID --job-id 1234 --cluster alpha
servatus resolve STATE_DIR ALLOCATION_ID --not-submitted
```

To extend an open Campaign through the CLI, pass the complete previously registered JSONL prefix
followed by the new suffix. Supplying only the suffix, changing the prefix, or extending after
`servatus seal` fails closed. Relative `stdin_file` paths resolve against `TASKS.jsonl`'s parent.
The CLI loads exactly `Path.cwd() / "SERVATUS.toml"`; `--profile NAME` overrides its declared
default. It never accepts a configuration path or searches a parent directory.

`PLAN.json` is published owner-only and never overwrites an existing path. It contains task keys,
requested resources, effective allocation totals, target values,
exact nonsecret `sbatch` arguments, allocation identities, and digests—not task arguments or stdin.
Complete scripts are shown only by the warning-bearing `--show-scripts` diagnostic. `validate`
makes one serial `sbatch --test-only` call per distinct allocation shape and prints each stable
shape/script digest plus the controller response. Its answer is time-specific and does not submit
or mutate campaign state.

An intent without a receipt is ambiguous. `reconcile` uses the Campaign's validated target lineage
for one bounded `squeue`/`sacct` query and adopts only one exact Servatus identity. Otherwise an
operator must resolve it explicitly as accepted or not submitted. Retry is explicit through
`Campaign.plan(..., retry={...})`; prior receipts remain in history. `inspect` reports
scheduler-only evidence as JSON. There is no acceptance-only status or caller-built completed set.

### Inspection

`Campaign.inspect()` returns one immutable, revision-bound view of the complete attempt history,
current allocation evidence, and optional caller-owned result evidence:

```python
def result_exists(task: Task) -> bool:
    path = Path("results") / task.key
    return path.is_file()


view = campaign.inspect(result_exists)
```

The plain synchronous probe runs exactly once per Task outside the Campaign lock. It returns `True`
only after validating one immutable or version-addressed canonical result, returns `False` for a
missing or incomplete result, and raises for invalid or untrustworthy content. Servatus stores
neither the probe nor its answer and never interprets an application schema. Omitting the probe
marks results unobserved. `scheduler=False` makes no scheduler call and explicitly leaves scheduler
evidence unobserved:

```python
result_view = campaign.inspect(result_exists, scheduler=False)
```

Scheduler inspection follows durable Attempt order. Each accepted Attempt gets one server-side
single-job `squeue` request and one `sacct --duplicates` request bound to its allocation-derived job
name. Queue rows must also repeat the exact allocation comment. Accounting rows may repeat that
comment or omit it, as some Slurm sites do; a different nonempty comment is unrelated evidence.
Exactly one accounting record submitted inside the reconciliation window anchors the original job.
Strictly later records with the same immutable identity are its requeue incarnations, and the unique
latest incarnation owns accounting evidence. Distinct Attempts remain distinct even if Slurm reuses
the same job ID and cluster. The native single-job invalid-ID response means only that no active row
exists, so accounting is still queried; without an accounting anchor, even an exact active row
reports `UNKNOWN`.

Each SSH command has a 30-second deadline. Local and remote argument vectors have at most 16
arguments, 16 KiB of complete command text, 1 MiB per output stream, 128 lines, and 4 KiB per source
field. Scheduler commands run with a fixed C locale and UTC timezone. A failed command, timeout,
overflow, malformed or partial row, unrelated Attempt identity, or ambiguous accounting history
aborts the whole inspection.

Allocation states normalize to `QUEUED`, `RUNNING`, `SUCCEEDED`, `FAILED`, `CANCELLED`, or
`UNKNOWN`. Packed Tasks share their allocation evidence. Every Attempt remains visible, while the
latest accepted Attempt owns a Task's current execution projection and unresolved acceptance
dominates that projection. `results_ready` means the roster is sealed and every result probe was
valid. `quiescent` independently requires requested scheduler evidence, no unresolved acceptance,
and terminal evidence for every accepted Attempt. Valid results can therefore be ready while Slurm
accounting is unknown. Inspection is transient, time-stamped, redacted, and read-only; a concurrent
Campaign revision change rejects the view.

A plan freezes the exact Campaign revision, roster and attempt projection, result and scheduler
observations, selection, retry and duplicate-risk choices, profile label and resolved values, and
allocations. `plan_document()` and `restore_plan()` provide its canonical cross-process codec.
Duplicate-risk warnings derive from the frozen acknowledgement keys rather than a second serialized
field.
Before each allocation submission, `Campaign.submit()` rereads Campaign state, reprobes only its
selected Tasks when the plan was result-aware, refreshes relevant accepted Attempts, rereads state,
and aborts changed eligibility before recording intent or calling `sbatch`. Pass the same probe as
`submit(plan, probe=result_exists)` for result-aware plans. Scheduler-only plans need no probe.

`Campaign.read_log(allocation_id, task_key=None, max_bytes=65_536)` returns one immutable
`LogSnapshot` containing the latest bounded suffix of an accepted Attempt's combined binary log.
Omit `task_key` for the allocation wrapper or supply a Task from that exact Attempt; Servatus derives
the accepted job ID, packed slot, target, and path. The byte limit must be from 1 byte through 1 MiB,
and `truncated` reports whether the remote read found one extra byte. Empty content is valid.

Log reads use one fixed bounded OpenSSH operation outside the Campaign lock. Missing, unreadable,
unavailable, overflowing, or otherwise untrustworthy logs raise a redacted `ObservationError`
without partial bytes. Log content is sensitive, untrusted binary data and may contain terminal
control sequences. Do not print it directly to a terminal; redirect CLI output to a private file or
use a safe binary viewer. Log content never enters Campaign state or views and cannot affect result
readiness, quiescence, planning, retry, reconciliation, or application validity. The remote log
namespace remains controlled by the same cluster account; Servatus does not prove a stable remote
inode or authenticate content after Slurm writes the derived path.

### Operational record

`Campaign.record(view)` returns schema-1 canonical JSON bytes for one exact revision-bound view. A
stale, foreign, or altered view is rejected. The call is read-only and never publishes the bytes:

```python
view = campaign.inspect()
record = campaign.record(view)
Path("private/campaign-record.json").write_bytes(record)
```

The record contains its observation time and scheduler-observation mode; Campaign identity,
revision, sealed state, roster digest, and ordered Task keys; and ordered Attempt provenance. Each
Attempt retains its Profile label, Task/retry/duplicate-risk keys, target and resource lineage
digests, allocation shape, attempt/retry/plan/script digests, acceptance or resolution revision,
receipt Job ID and cluster, and normalized scheduler state, exit code, timestamps, and observation
time. The retry digest hashes the canonical ordered retry and duplicate-risk keys. The attempt digest
hashes the canonical immutable redacted intent: allocation identity, Campaign revision, Profile
label, ordered Task keys, retry digest, lineage digests, allocation shape, and plan/script digests.

Task arguments, stdin, per-Task digests, complete scripts, `sbatch` arguments, target values,
environment, raw scheduler states and reasons, result evidence, logs, checkpoints, metrics, and
application outputs are excluded. The record changes only with fields included in that projection.
It is redacted, not anonymous or secret-safe: Task keys, allocation identities, and Slurm Job IDs
can identify work. Keep Campaign state and operational records private unless a separate review
approves a narrower publication projection.

Servatus does not cancel jobs in V1. Use the receipt with the site's normal `scancel` command.
Cancellation applies to the packed allocation, does not prove application completion, and does not
enable retry automatically.

## Publication

Use `publish` when failed work is disposable:

```python
from pathlib import Path

from servatus import Draft, publish


def build(draft: Draft) -> None:
    (draft.path / "result.json").write_text('{"status":"complete"}\n')


publication = publish(Path("outputs/run-1"), build)
```

The builder owns draft contents and validation. It must finish all content mutations before
returning; Servatus syncs the quiescent draft afterward.

When the application has finished authoring an owner-only sibling tree, `publish` can retire that
tree only after the canonical destination is committed and synced:

```python
bundle = Path("outputs/.run-1.active")
publication = publish(Path("outputs/run-1"), build, retire=bundle)
```

The retained tree must already exist, differ from the destination, share its exact parent, and be
quiescent before the call. Servatus pins it before entering the builder. Builder, validation,
collision, and other precommit failures preserve it. After a durable commit, Servatus removes only
the pinned tree and syncs the parent again. If that exact removal or its durability cannot be
proved, publication still succeeds, `Publication.cleanup_pending` is true, and one best-effort
`RuntimeWarning` reports the residue.

Use `publish_file` for one canonical regular file:

```python
from pathlib import Path

from servatus import publish_file


def write(stage: Path) -> None:
    stage.write_text('{"status":"complete"}\n')
    # Perform application validation before returning.


publication = publish_file(Path("outputs/protocol.json"), write)
```

The writer receives an existing empty adjacent regular file. It must write and validate that inode
in place; unlinking, replacing, or changing its file type fails publication. Its initial mode is
created from `0o666` through the process umask, and an explicit writer `chmod` is preserved. The
writer must finish all content mutations before returning.

Use `Workspace` when a worker must retain private checkpoints across restarts:

```python
from servatus import Draft, Workspace

destination = Path("outputs/model-1")
with Workspace(destination, identity=b"model request bytes") as workspace:
    checkpoint = workspace.path / "last.ckpt"
    # The application creates or resumes its own checkpoint here.

    def assemble(draft: Draft) -> None:
        draft.link(checkpoint, "last.ckpt")
        # Perform application validation before returning.

    publication = workspace.publish(assemble)
```

`Workspace` binds stable hidden state to the SHA-256 digest of opaque identity bytes and holds a
nonblocking writer lock. `Draft.link` atomically hard-links the source path into a safe relative
path, then accepts only a same-filesystem regular file. The hard-link operation selects the source
inode, so a safe source-path replacement before that operation may be selected. The builder owns
contents, validation, schemas, and completion meaning and must finish mutating linked contents
before returning. The owner-only draft namespace must remain quiescent during each `Draft.link()`;
hostile same-account replacement of its destination leaf during that call is outside the contract.

Independent workers can publish resumable child results beneath one future destination without
entering the parent:

```python
parent = Workspace(Path("outputs/study-1"), identity=b"study request bytes")

with parent.child("method-0", identity=b"method request bytes") as child:
    checkpoint = child.path / "last.ckpt"
    # Create or resume application work, then retain one immutable child result.
    child.publish(lambda draft: draft.link(checkpoint, "result.bin"))

# After the application decides all required children are valid:
with parent as workspace:
    workspace.publish(
        lambda draft: draft.link(parent.path / "method-0/result.bin", "method-0/result.bin")
    )
```

`child()` accepts one safe leaf and opaque identity. Different children may run concurrently; the
same child and parent finalization remain exclusive and nonblocking. A failed child retains only
its resumable private work, while a published child becomes immutable input under the parent work.
Servatus does not track expected children, readiness, dependencies, or application completion.

The owner-only hidden Workspace container is the lifecycle trust root. Servatus pins and rechecks
its entries without following links; unsafe substitution is preserved and reported as pending
cleanup. Callers must protect the parent directory, run only trusted same-account code, and use a
filesystem with stable cross-client inode identities. See [SECURITY.md](SECURITY.md).

## Guarantees and support boundary

- Campaign files are bounded, owner-only, schema-versioned, symlink-safe, atomically replaced, and
  synced. Every durable read validates the complete snapshot. One tagged Attempt outcome prevents
  an allocation from being both accepted and resolved as not submitted; its revision preserves
  delayed-resolution chronology.
- Intent preserves the normalized route, guardrails, requested resources, exact allocation totals,
  and reviewed nonsecret `sbatch` command before external acceptance.
- A destination is absent or one complete regular file or directory. Native commits and the Linux
  regular-file fallback never overwrite an existing entry. An already-present destination is
  rejected before its callback; the commit remains no-replace against later races. The Linux
  directory fallback locks the parent only for its absence check, rename, and published-inode
  verification. The lock is released before the post-commit parent sync, so a stalled remote sync
  does not serialize publications to other destinations.
- Work, hard-link sources, stages, and destination must share a filesystem.
- Disposable stage names are not synced merely by creation. Files and directories are synced before
  commit; publication returns only after the parent is synced. Failure or interruption after rename
  can leave a complete visible destination whose directory durability is not yet proven.
- Builders and writers must be quiescent when they return. Servatus verifies pathname/inode
  identity and syncs content, but does not detect or exclude concurrent content writers.
- Builder failures expose no destination. Resumable work remains; disposable stages are removed.
- Successful workspace publication exactly removes its pinned private tree. A moved, substituted,
  or unremovable tree remains visible as cleanup residue and is reported separately.
- Directory publication can retire one exact owner-only sibling after commit. Precommit failure
  preserves it; unsafe or incomplete post-commit retirement sets `cleanup_pending` without changing
  publication success.
- Child workspaces share the parent lifecycle lease; parent publication is busy until they close.

Publication supports POSIX filesystems on Linux and macOS. Its Linux fallbacks require cooperating
Servatus publishers, coherent advisory locks, owner-controlled parents, and stable inode identities;
hardware durability still depends on the filesystem and mount. Campaign submission is an
unprivileged workstation-side OpenSSH client for homogeneous independent processes in one-node
Slurm allocations. See [SECURITY.md](SECURITY.md) and [ADR 0003](docs/adr/0003-native-slurm-campaign.md)
for the exact fallback and live-acceptance envelope.

## Non-goals

Servatus is not an ML framework, scheduler plugin, daemon, security boundary, experiment tracker,
DAG engine, secrets manager, or transfer/image-deployment tool. V1 has no Submitit or runtime Python
dependency, plugin/backend abstraction, local executor, arrays, heterogeneous tasks, multi-node
ranks, MPI/torchrun, fractional/shared GPUs, queue-aware packing, automatic retry, background
polling, cancellation/requeue, raw Slurm/environment passthrough, serialized probes, application
schemas, log parsing/following/caching, compatibility shims, or cross-filesystem copy fallback.

See the [context glossary](docs/CONTEXT.md) and [architecture decisions](docs/adr/README.md) for the
ownership boundary.
