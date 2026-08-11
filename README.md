# Servatus

Run resumable work through Slurm and atomically publish validated outputs.

Servatus 0.4.1 combines durable publication with the native Slurm Campaign interface below.

```sh
pip install servatus
```

## Campaigns

A Campaign freezes an ordered prefix of opaque tasks. Reopening with the exact sequence is
idempotent; reopening with that exact prefix plus a nonempty suffix durably registers the new
tasks. Removing, reordering, or changing any registered task fails. Planning is local and
deterministic. Submission records durable intent before contacting Slurm, records the acceptance
receipt afterward, and stops on an ambiguous missing receipt rather than risking duplicate work.

```python
from pathlib import Path, PurePosixPath

from servatus import Campaign, ResourceRequest, SlurmTarget, Task

campaign = Campaign.open(
    Path("state/training"),
    [Task("candidate-0", ("train", "--candidate", "0"), b'{"seed": 7}\n')],
)
resources = ResourceRequest(
    cpus_per_task=8,
    memory_mib_per_task=32768,
    gpus_per_task=1,
    time_limit="1-00:00:00",
)
target = SlurmTarget.from_toml(Path("TARGET.toml"))
plan = campaign.plan(target, resources)
# Review plan.allocations and plan.digest, then submit explicitly:
receipts = campaign.submit(plan)
```

`ResourceRequest` has no defaults. CPU and MiB memory are positive, GPUs are a nonnegative whole
count, and time uses canonical `[days-]hours:minutes:seconds`. One request applies to every Task in
one Campaign. CPU-only, one-GPU, and one-process whole-multi-GPU tasks are supported. A project uses
separate Campaigns for different resource shapes.

The request retains the authored wall time for provenance. Plans and `sbatch` show Slurm's effective
limit, rounded upward once to whole minutes. The same rounding is applied before comparing a request
with the target ceiling; packed task count never multiplies wall time.

`RESOURCES.toml` contains exactly four required values:

```toml
cpus_per_task = 32
memory_mib_per_task = 65536
gpus_per_task = 1
time_limit = "3-00:00:00"
```

`TARGET.toml` describes one concrete execution lane and its conservative guardrails:

```toml
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
```

Unknown keys, counted GRES, relative remote paths, unsafe site tokens, controls, booleans used as
integers, unlimited/zero resources, and conflicting GPU settings are rejected. A target profile is
a user-side mistake guard, not cluster authorization. Every listed partition must fit one truthful
conservative envelope.

The planner preserves authored order and uses the fewest balanced groups allowed by every declared
ceiling. An allocation containing `n` Tasks requests exactly `n*C` CPUs, `n*M` MiB, and `n*G` GPUs;
time remains `T`. A caller may lower packing with `tasks_per_allocation`, but a cap above feasible
capacity is rejected rather than clamped. Servatus never rounds up to node capacity.

Append-only growth preserves target/resource lineage, accepted receipts, retry history, and
ambiguous intents. It increments campaign revision, so a plan made before the append becomes stale.
Accepted prefix tasks are not selected again unless the caller explicitly requests retry.

The current development line uses Campaign and plan schema 3. Plans retain typed planning inputs
and derived allocation summaries; loading regenerates the immutable plan once and requires the
same canonical bytes. Campaign state stores resource lineage once, exact intent provenance, compact
acceptance receipts, and allocation IDs explicitly resolved as not submitted. Derivable command
digests, lineage copies, job names, receipt Task keys, and negative-resolution wrappers are not
persisted. Schema 2 state and plans are rejected rather than interpreted through a compatibility
path; create a new Campaign when upgrading.

Each allocation runs one concurrent
`srun --exclusive --exact --nodes=1 --ntasks=1` step per Task. Each step receives its exact CPU,
MiB, and whole-GPU request and starts the target's immutable Apptainer image from `work_root`.
CPU-only work emits no GRES or `--nv`. Servatus never emits job-level exclusivity, overlap, all
memory, manual CUDA indices, ranks, or raw scheduler flags.

Slurm writes combined allocation stdout/stderr to `log_root/%j.out` and each combined task stream
to `log_root/%j-<zero-based-slot>.out`. `%j` is expanded by Slurm after it assigns the job ID; plans
and durable intent therefore remain immutable before scheduler acceptance.

Servatus requests concurrent exact steps; actual simultaneous placement depends on the site's CPU
and GRES topology and a truthful `ResourceRequest` and target profile. On an SMT2 site, one Slurm
CPU may represent one logical thread while an exclusive step occupies a physical core, so one
requested CPU can account for only half the logical capacity needed by that step. The accepted
four-step production smoke therefore used `cpus_per_task=2`. Servatus does not silently inflate CPU
requests, disable binding, or expose raw scheduler flags.

Task arguments and byte-exact stdin are embedded in the complete batch script before `sbatch`
acceptance. They are excluded from ordinary plan and status output, but are not secrets: cluster
administrators and accounting systems may be able to inspect them. Scheduler names expose only a
random Servatus allocation identity.

### CLI

The Python interface is authoritative. The CLI task JSONL adapter has exactly `key`, string-array
`args`, and `stdin_file` per line:

```sh
servatus plan TASKS.jsonl --target TARGET.toml --resources RESOURCES.toml \
  --campaign STATE_DIR --output PLAN.json --tasks-per-allocation 4
# Explicit sensitive diagnostic; prints complete scripts, arguments, and payloads:
servatus plan TASKS.jsonl --target TARGET.toml --resources RESOURCES.toml \
  --campaign STATE_DIR --output PLAN.json --show-scripts
servatus validate STATE_DIR PLAN.json
servatus submit STATE_DIR PLAN.json
servatus status STATE_DIR
servatus reconcile STATE_DIR ALLOCATION_ID --target TARGET.toml
servatus resolve STATE_DIR ALLOCATION_ID --job-id 1234 --cluster alpha
servatus resolve STATE_DIR ALLOCATION_ID --not-submitted
```

To extend an existing Campaign through the CLI, pass the complete previously registered JSONL
prefix followed by the new suffix. Supplying only the suffix or changing the prefix fails closed.

`PLAN.json` contains task keys, requested resources, effective allocation totals, target values,
exact nonsecret `sbatch` arguments, allocation identities, and digests—not task arguments or stdin.
Complete scripts are shown only by the warning-bearing `--show-scripts` diagnostic. `validate`
makes one serial `sbatch --test-only` call per distinct allocation shape and prints each stable
shape/script digest plus the controller response. Its answer is time-specific and does not submit
or mutate campaign state.

An intent without a receipt is ambiguous. `reconcile` performs one bounded `squeue`/`sacct` query
and adopts only one exact Servatus identity. Otherwise an operator must resolve it explicitly as an
accepted job or as not submitted. This is fail-closed recovery, not exactly-once execution. Retry is
explicit through `Campaign.plan(..., retry={...})`; prior receipts remain in history and resources
cannot change. Scheduler acceptance never means application completion: the caller supplies
`completed` after its own canonical validation.

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
created from `0o666` through the process umask, and an explicit writer `chmod` is preserved.

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
nonblocking writer lock. `Draft.link` only hard-links regular files into a safe relative path. The
application must not mutate a linked source inode after `Draft.link()` returns and before
publication completes. It owns contents, validation, schemas, and completion meaning.

If Linux installs a Workspace identity through the regular-file fallback but cannot prove both stage
removal and parent-directory durability, the valid identity remains authoritative and Servatus emits
a `RuntimeWarning` that identity-stage cleanup remains pending. An absent stage triggers one parent
sync retry before Servatus reports pending cleanup.

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

The owner-only hidden Workspace container is the lifecycle trust root. Within an authentic
container, Servatus requires the container, work directory, lifecycle lock, and identity file to
belong to the effective user with no group or world permissions. It rechecks those properties with
the pinned entries before application access, publication, and cleanup. First initialization syncs
the lock and work entries, container, and destination parent before committing identity. Cleanup
walks the pinned tree descriptor-relatively without following links and removes the root only while
its name still denotes the pinned inode; a moved or substituted name is preserved and reported as
pending cleanup. Servatus is not a defense against arbitrary same-account code renaming and
recreating the entire trust root. Callers must protect its parent directory and run only trusted
workers and builders. Workspace identity records store only exact inode pins for the container,
lock, and work entries. Cross-client reopen requires those inode identities to remain stable; live
opens still enforce local entry type, same-filesystem placement, and pathname-to-inode identity.

## Guarantees and support boundary

- Campaign files are bounded, owner-only, schema-versioned, symlink-safe, atomically replaced, and
  synced. Every durable read validates the complete snapshot, including the rule that one
  allocation cannot be both accepted and resolved as not submitted.
- Intent preserves the normalized route, guardrails, requested resources, exact allocation totals,
  and reviewed nonsecret `sbatch` command before external acceptance.
- A destination is absent or one complete regular file or directory. Native commits and the Linux
  regular-file fallback never overwrite an existing entry. The Linux directory fallback serializes
  cooperating Servatus publishers with an exclusive parent-directory lock.
- Work, hard-link sources, stages, and destination must share a filesystem.
- Disposable stage names are not synced merely by creation. Files and directories are synced before
  commit; the parent is synced after publication.
- Builder failures expose no destination. Resumable work remains; disposable stages are removed.
- Successful workspace publication exactly removes its pinned private tree. A moved, substituted,
  or unremovable tree remains visible as cleanup residue and is reported separately.
- Child workspaces share the parent lifecycle lease; parent publication is busy until they close.

Publication supports POSIX filesystems on Linux and macOS. Linux first uses
`renameat2(RENAME_NOREPLACE)`. If the kernel or filesystem reports only `EINVAL`, `ENOSYS`, or
`EOPNOTSUPP`, regular files use an atomic same-directory hard link followed by stage removal.
Directories use a descriptor-relative rename while holding an exclusive advisory lock on a pinned,
owner-controlled parent. That fallback requires every same-account publisher on every client to use
Servatus and the filesystem mount to provide one coherent `flock` domain and stable inode identities
across those clients. Local-only or disabled lock modes, unstable cross-client inodes, and group- or
world-writable parents are unsupported. Hardware durability still depends on the filesystem and
mount. Campaign submission is an unprivileged workstation-side OpenSSH client for
homogeneous independent processes in one-node Slurm allocations. It invokes the target's absolute
Slurm and Apptainer paths and uses a minimal sanitized scheduler environment.

The 0.1.0 client was live-validated on Slurm 23.11.4 with `select/cons_tres` `CR_CPU_MEMORY`, task
cgroup and affinity plugins, and absolute OpenSSH, Slurm, and Apptainer executables. The accepted
envelope covered CPU-only, one-GPU, one-process/two-GPU, and four packed one-GPU tasks. This is a
tested envelope, not a claim that editable target files enforce cluster policy or that other site
topologies preserve simultaneous placement. See ADR 0003 for the concise acceptance record.

## Non-goals

Servatus is not an ML framework, scheduler plugin, daemon, security boundary, experiment tracker,
DAG engine, secrets manager, or transfer/image-deployment tool. V1 has no Submitit or runtime Python
dependency, plugin/backend abstraction, local executor, arrays, heterogeneous tasks, multi-node
ranks, MPI/torchrun, fractional/shared GPUs, queue-aware packing, automatic retry, background
polling, cancellation/requeue, raw Slurm/environment passthrough, application completion probes,
compatibility shims, or cross-filesystem copy fallback.

See the [context glossary](docs/CONTEXT.md) and [architecture decisions](docs/adr/README.md) for the
ownership boundary.
