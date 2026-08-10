# Servatus

Run resumable work through Slurm and atomically publish validated outputs.

Servatus 0.1.0 combines durable publication with the native Slurm Campaign interface below.

```sh
pip install servatus
```

## Campaigns

A Campaign freezes an ordered set of opaque tasks. Planning is local and deterministic. Submission
records durable intent before contacting Slurm, records the acceptance receipt afterward, and stops
on an ambiguous missing receipt rather than risking duplicate work.

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

Each allocation runs one concurrent
`srun --exclusive --exact --nodes=1 --ntasks=1` step per Task. Each step receives its exact CPU,
MiB, and whole-GPU request and starts the target's immutable Apptainer image from `work_root`.
CPU-only work emits no GRES or `--nv`. Servatus never emits job-level exclusivity, overlap, all
memory, manual CUDA indices, ranks, or raw scheduler flags.

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

## Guarantees and support boundary

- Campaign files are owner-only, schema-versioned, symlink-safe, atomically replaced, and synced.
- Intent preserves the normalized route, guardrails, requested resources, exact allocation totals,
  and reviewed nonsecret `sbatch` command before external acceptance.
- A destination is absent or one complete directory; it is never overwritten.
- Work, hard-link sources, stages, and destination must share a filesystem.
- Files and directories are synced before a kernel-exclusive commit; the parent is synced after.
- Builder failures expose no destination. Resumable work remains; disposable stages are removed.
- Successful workspace publication removes private state. Cleanup residue is reported separately.

Publication supports POSIX filesystems on Linux and macOS. Hardware durability still depends on the
filesystem and mount. Campaign submission is an unprivileged workstation-side OpenSSH client for
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
