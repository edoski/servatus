# Servatus

Run resumable work through Slurm and atomically publish validated outputs.

Servatus 0.9.0 provides a native Slurm campaign interface and durable POSIX publication.

```sh
pip install servatus
```

## Campaigns

A Campaign owns an ordered roster of opaque tasks and its durable execution history. Create a fixed
roster, load its execution profile, and review a plan before submitting:

```python
from pathlib import Path

from servatus import Campaign, Profile, Task

campaign = Campaign.create(
    Path("training-state"),
    [Task("candidate-0", ("train", "--candidate", "0"), b'{"seed": 7}\n')],
)
profile = Profile.load(Path("SERVATUS.toml"), name="research")
plan = campaign.plan(profile)
# Review plan.allocations and plan.digest before submitting.
# validation = campaign.validate(plan)  # Optional Slurm --test-only check.
result = campaign.submit(plan)
# Later processes load the durable roster without resupplying Tasks:
campaign = Campaign.load(Path("training-state"))
```

Creation seals the roster by default. Use `appendable=True` when more work will arrive, then call
`campaign.append(new_tasks)` with only the new suffix. Registered task keys, arguments, bytes, and
order never change. `campaign.tasks` returns the immutable authored tuple. `campaign.seal()` ends
authoring irreversibly and is idempotent. Append and seal advance the revision and invalidate older
plans; an appendable campaign may execute before sealing.

`ResourceRequest` has no defaults. CPU and MiB memory are positive, GPUs are a nonnegative whole
count, and time uses canonical `[days-]hours:minutes:seconds`. One request applies to every Task in
a plan. A later plan may use a different target or resource request; each Attempt retains its
original resolved configuration. The application owns whether changing an image or work root
preserves Task meaning; storing a path does not pin its contents. CPU-only, one-GPU, and
one-process whole-multi-GPU tasks are supported.

The request retains authored wall time. Plans and `sbatch` use Slurm's effective limit, rounded
upward once to whole minutes. The same rounding applies before comparing a request with the target
ceiling; packed task count never multiplies wall time.

`SERVATUS.toml` contains named execution profiles and an optional `default_profile`. An explicit
name overrides the default. Unknown keys and malformed TOML are rejected throughout the document;
only the selected profile's resource and target values undergo semantic validation.

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

The selected profile rejects counted GRES, relative remote paths, unsafe site tokens, controls,
booleans used as integers, unlimited/zero resources, incomplete values, and conflicting GPU settings.
Profiles do not inherit, merge, search parent directories, consult environment variables, or use a
global store. Labels are provenance; resolved values govern execution. A target is a user-side
mistake guard, not cluster authorization. Every listed partition must fit one truthful conservative
envelope.

The planner preserves authored order and uses balanced groups within every declared ceiling. An
allocation containing `n` Tasks requests exactly `n*C` CPUs, `n*M` MiB, and `n*G` GPUs; time remains
`T`. A caller may lower packing with `tasks_per_allocation`, but a cap above feasible capacity is
rejected. Servatus never rounds requests up to node capacity. Each reviewed batch contains at most
`max_allocations_per_submit` allocations. The plan exposes `deferred_task_keys` for eligible work
beyond that batch and `excluded_task_keys` for work withheld by the eligibility policy. Neither is
silently submitted.

`campaign.plan(profile, probe=None, retry=(), allow_duplicate_risk=(), tasks_per_allocation=None)`
collects current evidence itself. Valid application results are excluded. Never-accepted missing or
unobserved Tasks are eligible. Unresolved acceptance and active or held work block retry. Terminal
accepted work requires explicit retry; unknown accepted work also requires explicit duplicate-risk
acknowledgement. Every historical Attempt participates in this decision, using its original route.

A plan is a compact reviewed decision bound to a Campaign revision. It stores selected allocations,
resolved execution configuration, retry choices, whether a result probe is required, and one
integrity digest. `plan_document()` and `restore_plan()` provide its cross-process codec; restoration
performs no scheduler or probe calls. Transient observations remain outside the serialized plan.
Task arguments and stdin remain in private Campaign state, outside the plan document. Campaign
and plan documents use schema 5; unsupported schemas are rejected without migration.

Before each allocation, submission refreshes relevant scheduler evidence and, for result-aware
plans, probes its selected Tasks again. Supply the probe as `campaign.submit(plan, probe=probe)`.
Submission checks local command bounds before claiming work, atomically records durable intent,
then contacts Slurm outside the state lock. Unresolved intent conservatively blocks duplicate work.

`submit()` returns a structured `SubmitResult`: `receipts` contains confirmed receipts,
`unresolved` contains uncertain submissions, and `unattempted` contains reviewed allocations left
untouched when an operational failure or concurrent change stops the batch. `stop_reason` explains
the stop, or is `None` when the whole batch completes. If Slurm accepts work but receipt persistence
fails, the receipt appears as `observed_receipt` on the corresponding unresolved submission,
separately from durably recorded receipts. Save that evidence and reconcile before retrying.
`KeyboardInterrupt` and `SystemExit` propagate while preserving durable intent. A concurrent append
or seal stops further submission from the old plan but cannot discard an already observed receipt. Recording an
identical receipt is idempotent; conflicting outcomes fail.

### CLI

The task JSONL adapter accepts exactly `key`, string-array `args`, and `stdin_file` per line.
Relative input paths resolve against the JSONL file's parent.

```sh
servatus create STATE_DIR TASKS.jsonl
# For a growing roster, create with --appendable, then append only new Tasks:
servatus create GROWING_STATE INITIAL.jsonl --appendable
servatus append GROWING_STATE NEW_TASKS.jsonl
servatus seal GROWING_STATE

servatus plan STATE_DIR --profile research --output PLAN.json --tasks-per-allocation 4
servatus plan STATE_DIR --profile research --output RETRY.json --retry task-0
# Unknown accepted work requires the separate duplicate-risk acknowledgement:
servatus plan STATE_DIR --profile research --output RISKY.json \
  --retry task-0 --allow-duplicate-risk task-0
servatus validate STATE_DIR PLAN.json
servatus submit STATE_DIR PLAN.json
servatus status STATE_DIR
servatus logs STATE_DIR ALLOCATION_ID --task TASK_KEY --bytes 65536 > task.log
servatus reconcile STATE_DIR ALLOCATION_ID
servatus resolve STATE_DIR ALLOCATION_ID --job-id 1234 --cluster alpha
servatus resolve STATE_DIR ALLOCATION_ID --not-submitted
```

Planning loads exactly `Path.cwd() / "SERVATUS.toml"`; `--profile NAME` overrides its declared
default. The CLI never searches a parent directory or accepts a configuration path. Planning does
not create or extend Campaign state. `PLAN.json` is published owner-only without overwrite.
Complete scripts, arguments, and payloads require the explicit sensitive `--show-scripts` diagnostic.
`validate` makes serial `sbatch --test-only` calls for distinct allocation shapes. Validation is
time-specific and does not submit or mutate state. `submit` prints the full structured outcome and
returns exit status 1 when the batch stops early; a completed batch returns 0.

An intent without a receipt is ambiguous. `reconcile` uses that Attempt's original target and
bounded scheduler queries, adopting only an exact matching Servatus identity. Otherwise an operator
must explicitly resolve the Attempt as accepted or not submitted. Retry never erases earlier
Attempts. Cancellation uses the site's normal `scancel` command and applies to the packed allocation;
it does not prove application completion or enable retry automatically.

### Execution and inspection

Each allocation runs concurrent `srun --exclusive --exact --nodes=1 --ntasks=1` steps, one per Task.
Each step receives its exact CPU, MiB, and whole-GPU request and starts the target's immutable
Apptainer image from `work_root`. CPU-only work emits no GRES or `--nv`. GPU steps forward Slurm's
step-local `CUDA_VISIBLE_DEVICES` into Apptainer with `CUDA_DEVICE_ORDER=PCI_BUS_ID`; missing
visibility fails the step. Servatus emits no job-level exclusivity, overlap, manual CUDA indices,
ranks, or raw scheduler flags. Site configuration owns isolation and simultaneous placement.

Arguments and byte-exact stdin are embedded in the batch script before acceptance. Before starting
siblings, the batch decodes every payload into checked, owner-only files under private
`${TMPDIR:-/tmp}` storage. Workers may close stdin early. The batch waits for every started sibling,
aggregates failures, and removes payload files after completion or handled interruption. Compute
nodes need writable scratch and standard POSIX tools; no remote Python runtime is required. Inputs
are not secrets: cluster administrators and accounting systems may inspect them.

Slurm writes combined allocation stdout/stderr to `log_root/<allocation_id>-%j.out` and each task's
combined stream to `log_root/<allocation_id>-%j-<zero-based-slot>.out`. Slurm expands `%j`; the
immutable allocation identity prevents reused job numbers from aliasing Attempts.

`campaign.inspect()` returns transient, time-stamped, revision-bound diagnostics over all Attempts,
current task execution, and optional caller-owned results:

```python
def result_exists(task: Task) -> bool:
    # Validate an immutable or version-addressed canonical result here.
    return (Path("results") / task.key).is_file()


view = campaign.inspect(result_exists)
result_view = campaign.inspect(result_exists, scheduler=False)
```

The synchronous probe runs once per Task outside the lock. Return `True` only for a validated
canonical result, `False` for missing or incomplete content, and raise for invalid or untrustworthy
content. Servatus stores neither callback nor answers. Omitting the probe leaves results
unobserved; `scheduler=False` leaves scheduler evidence unobserved and makes no scheduler calls.

Inspection batches at most 16 distinct jobs per target/cluster into bounded `squeue` and
`sacct --duplicates` requests; reused job numbers are queried separately. Rows must match job number
and immutable allocation name. Queue rows also need the exact allocation comment; accounting may
omit it where site policy does. An original accounting record within the submission window anchors
later requeue incarnations. Contradictory identity or history fails closed. Positive queue evidence
blocks retry even when accounting has no anchor or reports a later terminal sample. Held/requeued
work, including `SPECIAL_EXIT`, is retained work and cannot prove quiescence.

Every SSH operation has a 30-second deadline and concurrently drains stdout/stderr. Local failures
kill and reap the child and close its streams. Commands permit at most 32 arguments, 16 KiB of
command text, 4 KiB per source field, and 1 MiB per output stream. Scheduler responses permit at most
128 lines per batch. Commands use a fixed C locale and UTC timezone. Timeout, overflow, malformed
rows, unrelated identities, and ambiguous accounting abort inspection without partial evidence.

Allocation states normalize to `QUEUED`, `RUNNING`, `SUCCEEDED`, `FAILED`, `CANCELLED`, or `UNKNOWN`.
Packed Tasks share allocation evidence. The latest accepted Attempt owns current execution, while
unresolved acceptance dominates it. `results_ready` requires a sealed roster and valid results for
every Task. Independently, `quiescent` requires scheduler evidence, no unresolved acceptance, and
terminal evidence for every accepted Attempt. A valid application result never proves a job stopped.
A concurrent Campaign revision change rejects inspection.

`campaign.read_log(allocation_id, task_key=None, max_bytes=65_536)` returns a bounded binary
`LogSnapshot` from the accepted Attempt's original route and derived log path. The maximum is
1 MiB; `truncated` reports whether more bytes exist. Remote failures raise a redacted
`ObservationError` without partial bytes. Logs are sensitive, untrusted data: redirect CLI output
to a private file or use a safe binary viewer. They have no authority over result validity, retry,
readiness, or quiescence. Servatus does not authenticate content in the account-owned remote log
namespace.

### Operational record

`campaign.record(view)` returns a redacted JSON projection for current diagnostics without writing
or publishing it:

```python
view = campaign.inspect()
Path("private/campaign-record.json").write_bytes(campaign.record(view))
```

The record contains Campaign identity and revision, roster keys, Attempt chronology, selected
Profile labels, allocation shapes, receipts, and normalized scheduler evidence. It is diagnostic
data, not execution authority. Task arguments, stdin, scripts, target values, logs, result evidence,
and application outputs are excluded. Keys and job identities still identify work; keep records private.

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
Servatus checks destination absence again after acquiring lifecycle leases, before exposing private
work. Initialization syncs the private hierarchy and its parent before committing the identity,
without holding shared parent coordination during synchronization. Servatus does not track expected
children, readiness, dependencies, or application completion.

The owner-only hidden Workspace container is the lifecycle trust root. Servatus pins and rechecks
its entries without following links; unsafe substitution is preserved and reported as pending
cleanup. Callers must protect the parent directory, run only trusted same-account code, and use a
filesystem with stable cross-client inode identities. See [SECURITY.md](SECURITY.md).

## Guarantees and support boundary

- Campaign files are bounded, owner-only, schema-versioned, symlink-safe, atomically replaced, and
  synced. Every durable read validates the complete snapshot. One tagged Attempt outcome prevents
  an allocation from being both accepted and resolved as not submitted; its revision preserves
  delayed-resolution chronology.
- Intent preserves the original route, guardrails, requested resources, ordered Task keys, and
  explicit revision and timestamp before external acceptance. Allocation totals, commands, and
  reconciliation windows derive from those values.
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
- Builders and writers must be quiescent when they return and throughout private cleanup. Servatus verifies pathname/inode
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
DAG engine, secrets manager, or transfer/image-deployment tool. It has no Submitit or remote Python
dependency, plugin/backend abstraction, local executor, arrays, heterogeneous resources within one
allocation, multi-node ranks, MPI/torchrun, fractional/shared GPUs, queue-aware packing, automatic
retry, background polling, cancellation/requeue, raw Slurm/environment passthrough, serialized probes, application
schemas, log parsing/following/caching, compatibility shims, or cross-filesystem copy fallback.

See the [context glossary](docs/CONTEXT.md) and [architecture decisions](docs/adr/README.md) for the
ownership boundary.
