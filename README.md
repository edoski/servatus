# Servatus

Durable Slurm sweeps from a laptop or a login node, with atomic publication of their results.

Servatus keeps one durable record of a sweep: which Tasks exist, which allocations were submitted,
and what Slurm accepted. It records its intent before it contacts the scheduler, so a dropped SSH
connection never silently doubles work. It tells you which Tasks are running, finished, failed, or
uncertain, and it never retries anything on its own. Inside the job, workers keep resumable
checkpoints in a private workspace and publish results atomically: a result directory is either
absent or complete.

Servatus has no runtime dependencies and needs no Python on the cluster to submit work. It talks to
Slurm through plain OpenSSH, or runs the Slurm commands directly when you are on a login node.

```sh
pip install servatus      # Python 3.11+, Linux or macOS
```

Workers that publish results also need `servatus` where they run, for example in the image.

## Quickstart: a sweep in 10 minutes

### 1. Describe the cluster

Put a `SERVATUS.toml` next to your launcher. Document-level `[target]` and `[resources]` tables are
shared defaults; each `[profiles.NAME]` overrides them key by key, so a CPU profile beside this one
needs only its own `resources`.

```toml
default_profile = "a100"

[target]
host = "hpc"                                # ssh_config alias; omit to run Slurm locally
slurm_bin = "/usr/bin"
work_root = "/scratch/alice/sweep"          # each Task starts here
log_root = "/scratch/alice/sweep/logs"
apptainer = "/usr/bin/apptainer"            # omit apptainer/image to run args[0] directly
image = "/scratch/alice/images/train.sif"
binds = ["/datasets:/datasets:ro"]
partitions = ["gpu"]
max_tasks_per_allocation = 4
max_cpus_per_allocation = 64
max_memory_mib_per_allocation = 262144
max_time_limit = "2-00:00:00"

[profiles.a100.target]
gpu_gres = "gpu:a100"
max_gpus_per_allocation = 4

[profiles.a100.resources]
cpus = 16
memory_mib = 65536
gpus = 1
time_limit = "12:00:00"
signal_before_end = "0:10:00"               # SIGUSR1 ten minutes before the limit
```

Create `work_root`, `log_root`, and `work_root/results` on the cluster once. `servatus doctor`
checks that the configuration loads and that the scheduler answers.

### 2. Write the worker

Each Task runs one process in its own `srun` step. The process gets its Task's arguments, stdin
bytes, and environment, plus `SERVATUS_TASK_KEY`, `SERVATUS_ALLOCATION_ID`, `SERVATUS_SLOT`,
`SERVATUS_JOB_ID`, and `SERVATUS_RESTART_COUNT`.

```python
# train.py, in work_root
import os, signal, sys, threading
from pathlib import Path

from servatus import Workspace
from servatus.errors import DestinationExists

from mylab import Trainer  # your code

stopping = threading.Event()
signal.signal(signal.SIGUSR1, lambda *_: stopping.set())

config = sys.stdin.buffer.read()  # the Task's stdin bytes
destination = Path("results") / os.environ["SERVATUS_TASK_KEY"]


def main() -> int:
    try:
        with Workspace(destination, identity=config) as workspace:
            checkpoint = workspace.path / "checkpoint.pt"
            trainer = Trainer.resume(checkpoint) if checkpoint.exists() else Trainer(config)
            while not trainer.done:
                trainer.step()
                if stopping.is_set() or trainer.step_count % 500 == 0:
                    trainer.save(checkpoint)  # write a temp file, then os.replace
                if stopping.is_set():
                    return 1  # out of time; the checkpoint survives
            trainer.save(checkpoint)

            def assemble(draft):
                draft.link(checkpoint, "model.pt")
                (draft.path / "metrics.json").write_text(trainer.metrics_json())

            workspace.publish(assemble)
    except DestinationExists:
        pass  # an earlier run already published it
    return 0


sys.exit(main())
```

The `Workspace` directory is bound to the identity bytes and survives failure, preemption, and
requeue. When Slurm requeues the job, or you resubmit the Task later, the worker reopens the same
private directory and resumes. `publish` commits `results/<key>` atomically and only then removes
the private work.

### 3. Define the Tasks

Either a JSONL file (one Task per line; `stdin_file` and `env` are optional, and relative paths
resolve against the JSONL file's directory):

```json
{"key": "lr-1e-3", "args": ["python", "train.py"], "stdin_file": "configs/lr-1e-3.json"}
{"key": "lr-3e-4", "args": ["python", "train.py"], "stdin_file": "configs/lr-3e-4.json", "env": {"WANDB_MODE": "offline"}}
```

```sh
servatus ensure state tasks.jsonl      # create the campaign, or append Tasks it has not seen
```

Or a Python launcher. Run it where it can see the results directory: on the login node (with
`host` omitted), or on a machine that mounts the cluster filesystem.

```python
import json
from pathlib import Path

from servatus import Campaign, Profile, Task

RESULTS = Path("/scratch/alice/sweep/results")


def finished(tasks):  # the probe: keys whose result is valid
    return {task.key for task in tasks if (RESULTS / task.key).is_dir()}


tasks = [
    Task(
        f"lr-{lr}-seed-{seed}",
        ["python", "train.py"],
        stdin=json.dumps({"lr": lr, "seed": seed}, sort_keys=True).encode(),
    )
    for lr in (1e-3, 3e-4, 1e-4)
    for seed in range(3)
]
campaign = Campaign.ensure("state", tasks, probe=finished)
plan = campaign.plan(Profile.load("SERVATUS.toml"))
print(f"{len(plan.selected)} selected, {len(plan.held)} held, {len(plan.deferred)} deferred")
result = campaign.submit(plan)
```

`ensure` is safe to rerun: existing Tasks must be identical by key (so build them
deterministically), and new keys are appended in order. With a probe, Tasks whose results already
exist are never resubmitted.

### 4. Plan, submit, watch

```sh
servatus plan state --output plan.json     # prints selected, held, and deferred Tasks
servatus submit state plan.json
servatus status state
servatus logs state --task lr-1e-3 --output lr-1e-3.log
servatus plan state --retry-failed --output retry.json    # later: rerun failed or cancelled Tasks
```

A plan is a reviewed decision. `submit` rebuilds it from the current campaign and refuses it if
anything changed since you reviewed it.

## Concepts

A **Campaign** is a directory holding an ordered roster of **Tasks** and every **Attempt** to run
them. A **Profile** pairs a **Target** (where and how to run, plus ceilings that guard against
mistakes) with **Resources** (what each Task needs). A **Plan** packs eligible Tasks into
single-node **allocations**, each running its Tasks as concurrent steps. Before contacting Slurm,
submission durably records an **intent**; Slurm's job number becomes a **Receipt**. Acceptance is
not completion: Servatus observes `squeue` and `sacct` to learn what happened, and an optional
**result probe** tells it which results are valid. Tasks that cannot run now are **held** with a
reason. The full glossary is in
[docs/CONTEXT.md](https://github.com/edoski/servatus/blob/main/docs/CONTEXT.md).

| Hold | Meaning | What to do |
| --- | --- | --- |
| `VALID` | The probe reports a valid result. | Nothing. |
| `ACTIVE` | Slurm still holds earlier work (queued, running, held, requeued). | Wait, or `cancel`. |
| `UNRESOLVED` | An intent has no recorded outcome. | See the recovery playbook. |
| `FINISHED` | Earlier work is terminal and no valid result is known. | `--retry KEY` (or `--retry-failed` if it failed). |
| `UNOBSERVABLE` | Slurm's evidence for earlier work is unknown. | `--retry KEY --allow-duplicate-risk KEY`. |
| `NOT_REQUESTED` | Excluded by `--only`. | Nothing. |

Eligible Tasks beyond the target's `max_allocations_per_submit` are **deferred** to the next plan.

## Recovery playbook

| Exit | Meaning | Next step |
| --- | --- | --- |
| 0 | Success. | |
| 1 | Error: bad input, refused plan, stale plan, conflicting or corrupt state. | Read the message. `StalePlan`: plan again. |
| 2 | Usage error. | Check `servatus COMMAND --help`. |
| 3 | Submission interrupted. Some allocations may be unresolved. | Run `status`, then resolve them as below. |
| 75 | Cluster unavailable or campaign busy. | Retry later. Submission checks the connection before recording anything. |

An **unresolved allocation** has a durable intent but no receipt: the `sbatch` call may or may not
have reached Slurm. Servatus holds its Tasks until you resolve it.

1. `servatus reconcile state ALLOCATION_ID` asks Slurm for a job carrying that allocation's
   identity. If exactly one matches, its receipt is recorded.
2. If reconcile cannot prove it, check `squeue`/`sacct` yourself. If you find the job, record it
   with `servatus mark-accepted state ALLOCATION_ID JOB_ID` (add `--cluster NAME` under
   federation). If you are sure it never reached Slurm, run
   `servatus mark-not-submitted state ALLOCATION_ID`.
3. Plan again. History is never erased; a new Attempt is added.

`servatus cancel state --task KEY` (or `--allocation ID`) runs `scancel` for the Task's current
allocation. Cancelling a packed allocation stops all its Tasks. Cancellation does not retry
anything; once Slurm reports the work cancelled, `--retry-failed` selects it.

## Publication

Publication works on its own, with or without a Campaign. Work, sources, stages, and destinations
must share one filesystem; Servatus never copies.

**`publish(destination, build, *, retire=None, mode=None)`** builds a directory in a private stage
and commits it with a no-replace rename. The builder receives a `Draft`; it writes into
`draft.path`, validates, and returns. The draft is invalid after the builder returns.

```python
from servatus import publish


def build(draft):
    (draft.path / "summary.json").write_text('{"ok": true}\n')
    draft.link_tree("runs/42/plots", "plots")  # hard-link every regular file under a tree


publish("outputs/run-42", build, mode=0o755)
```

**`publish_file(destination, write, *, mode=None)`** publishes one regular file. The writer
receives `<private stage>/<destination name>` with the real file name and suffix and may create it
any way it likes, including libraries that write a temporary file and rename it:

```python
import numpy as np
from servatus import publish_file

publish_file("outputs/weights.npy", lambda path: np.save(path, weights))
```

**`Workspace(destination, *, identity)`** is private, resumable work bound to opaque identity bytes
(see the worker above). `workspace.publish(build, *, mode=None)` publishes and then removes the
private work; `workspace.discard()` removes it without publishing. If the destination already
exists, entering the workspace removes any leftover private work and raises `DestinationExists`.
Entering work bound to a different identity raises `WorkspaceConflict`, naming its path.

**Children.** `parent.child(name, identity=...)` gives concurrent workers their own resumable
results beneath one future destination. Children run concurrently; the parent publishes once the
application decides they are all valid:

```python
parent = Workspace("outputs/study", identity=b"study v1")
with parent.child("method-a", identity=b"method-a v1") as child:
    ...  # create or resume child.path / "result.bin"
    child.publish(lambda draft: draft.link(child.path / "result.bin", "result.bin"))
with parent as workspace:
    workspace.publish(lambda draft: draft.link_tree(workspace.path / "method-a", "method-a"))
```

**Modes.** Directories are built owner-only (0700) and set to `mode` just before commit; files
likewise. The default is the usual umask-derived mode (`0o777 & ~umask` for directories).

**Hard links alias.** `Draft.link` and `link_tree` publish the *same inode* as their source. A
later in-place rewrite of the source (opening it for writing rather than replacing it) changes the
published result too. Replace sources with a new file and `os.replace`, or write fresh files into
`draft.path`.

`retire=` removes one owner-only sibling tree after the destination is durably committed. If any
cleanup cannot be proved, publication still succeeds with `Publication.cleanup_pending` set.

## Configuration reference

`Profile.load(path, *, name=None)` reads `SERVATUS.toml`. An explicit name wins over
`default_profile`; a sole profile selects itself. Unknown keys are rejected everywhere. Durations
are `[D-]H:MM:SS`; time limits round up to whole minutes, as Slurm enforces them.

| `[target]` key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `host` | string | omitted | SSH destination (`[user@]host` or ssh_config alias). Omitted: run Slurm locally. |
| `slurm_bin` | absolute path | required | Directory holding `sbatch`, `squeue`, `sacct`, `scancel`. |
| `work_root` | absolute path | required | Working directory of every Task step. |
| `log_root` | absolute path | required | Slurm log directory (must exist). |
| `partitions` | array of strings | required | Partitions the job may use. |
| `max_tasks_per_allocation` | int | required | Packing ceiling. |
| `max_cpus_per_allocation` | int | required | CPU ceiling per allocation. |
| `max_memory_mib_per_allocation` | int | required | Memory ceiling per allocation. |
| `max_time_limit` | duration | required | Wall-time ceiling. |
| `account`, `qos`, `constraint` | string | omitted | Passed to `sbatch` when set. |
| `gpu_gres` | string | omitted | Count-free GPU GRES (`gpu`, `gpu:a100`). Omit for CPU-only targets. |
| `max_gpus_per_allocation` | int | 0 | Must be positive exactly when `gpu_gres` is set. |
| `max_allocations_per_submit` | int | no cap | Allocations per plan; the rest are deferred. |
| `max_script_bytes` | int | 1048576 | Batch script size bound. |
| `apptainer`, `image` | absolute path | omitted | Apptainer launcher. Set both, or neither for the direct launcher. |
| `binds` | array of strings | `[]` | Extra Apptainer binds, `SRC[:DST[:ro\|rw]]`; `work_root` is always bound. |

| `[resources]` key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `cpus` | int | required | CPUs per Task. |
| `memory_mib` | int | required | MiB per Task. |
| `time_limit` | duration | required | Wall time of the allocation. |
| `gpus` | int | 0 | Whole GPUs per Task. |
| `signal_before_end` | duration | omitted | Send SIGUSR1 to each Task this long before the limit. |

An allocation of `n` Tasks requests exactly `n` times the per-Task CPUs, memory, and GPUs; time
stays the same. In Python, the same values are `Target`, `Apptainer`, `Resources`, and `Profile`.

**Launchers.** With Apptainer, each step runs `apptainer exec` on the image with a clean
environment, the Task's `env`, and `work_root` plus `binds` mounted; GPU steps get `--nv` and
Slurm's step-local `CUDA_VISIBLE_DEVICES`. Without Apptainer, each step runs the Task's absolute
`args[0]` under `env -i` with the same variables. Either way, Task stdin and environment are written
into the batch script and are visible to cluster administrators: never put secrets in Tasks.

**SSH.** Servatus runs `ssh -T -o BatchMode=yes -o LogLevel=ERROR HOST`, so it never prompts.
Login banners and shell-startup output are ignored. Configure the host in `~/.ssh/config`:

```
Host hpc
    HostName login.cluster.example.edu
    User alice
    ControlMaster auto
    ControlPath ~/.ssh/control-%C
    ControlPersist 30m
```

A persistent master connection makes every Servatus call fast, and on sites that require
multi-factor login it lets you authenticate once (`ssh hpc true`) before running Servatus. Keys
must be available without a passphrase prompt, for example through `ssh-agent`.

## CLI reference

`STATE` is the campaign directory. Planning reads `./SERVATUS.toml` unless `--config PATH` is given.
Output is human-readable; `--json` gives machine output. `servatus --version` prints the version.

| Command | Purpose |
| --- | --- |
| `create STATE TASKS.jsonl [--appendable]` | Create a campaign; fails if it exists. Sealed unless `--appendable`. |
| `ensure STATE TASKS.jsonl` | Create, or append Tasks not yet registered; existing keys must match. |
| `append STATE TASKS.jsonl` | Append new Tasks to an appendable campaign. |
| `seal STATE` | End authoring irreversibly. |
| `plan STATE [--profile NAME] [--output FILE]` | Show selected, held, and deferred Tasks; save the plan (0600, never overwrites). |
| `  --retry KEY`, `--retry-failed` | Retry one Task, or every Task whose current work failed or was cancelled. |
| `  --allow-duplicate-risk KEY` | Retry a Task whose earlier work is unobservable. |
| `  --only KEY`, `--tasks-per-allocation N` | Restrict the plan; lower packing. |
| `  --show-scripts` | Print complete batch scripts (sensitive). |
| `validate STATE PLAN` | `sbatch --test-only` once per distinct allocation shape. |
| `submit STATE PLAN` | Submit the reviewed plan. Exit 3 if interrupted. |
| `status STATE [--offline] [--json]` | Task and allocation status; `--offline` skips Slurm. |
| `logs STATE --task KEY \| --allocation ID [--bytes N] [--output FILE]` | Bounded log tail; writes 0600; refuses a terminal. |
| `reconcile STATE ALLOCATION` | Resolve an unresolved allocation from Slurm evidence. |
| `mark-accepted STATE ALLOCATION JOB_ID [--cluster NAME]` | Record a job you found yourself. |
| `mark-not-submitted STATE ALLOCATION` | Record that an allocation never reached Slurm. |
| `cancel STATE [--task KEY] [--allocation ID]` | `scancel` the matching current allocations. |
| `doctor [--profile NAME]` | Check the configuration and scheduler connection. |

## Guarantees

- Intent is durable before Slurm is contacted. An uncertain submission blocks its Tasks until it
  is resolved; nothing is retried without an explicit decision. ([ADR 0005][0005])
- Submission executes exactly the reviewed plan, or refuses it as stale or tampered.
- Campaign state is owner-only, validated on every read, atomically replaced, and synced.
- Scheduler evidence is bounded, identity-checked, and read from each Attempt's original target.
  ([ADR 0003][0003])
- A published destination is absent or complete and is never overwritten. Content is synced before
  commit and the parent after it. ([ADR 0002][0002])
- Private work survives failure and is removed only after a durable publication. ([ADR 0004][0004])
- Application meaning stays with you: Servatus never sees schemas or decides completion.
  ([ADR 0001][0001])

[0001]: https://github.com/edoski/servatus/blob/main/docs/adr/0001-opaque-application-seam.md
[0002]: https://github.com/edoski/servatus/blob/main/docs/adr/0002-posix-workspace-publication.md
[0003]: https://github.com/edoski/servatus/blob/main/docs/adr/0003-native-slurm-campaign.md
[0004]: https://github.com/edoski/servatus/blob/main/docs/adr/0004-concurrent-child-workspaces.md
[0005]: https://github.com/edoski/servatus/blob/main/docs/adr/0005-campaign-engine.md

Read [SECURITY.md](https://github.com/edoski/servatus/blob/main/SECURITY.md) for the trust model and
[ADR 0006](https://github.com/edoski/servatus/blob/main/docs/adr/0006-clean-break-0.12.md) for the
current scope.

## Non-goals

- **No job arrays and no multi-node jobs.** Every allocation is one node. One Task may still use
  several GPUs on that node, for example `torchrun --standalone --nproc-per-node=4` with `gpus = 4`.
- **No automatic retry or background polling.** You run `status` and `plan` when you want to.
- **No scheduler plugins.** One native Slurm lane, over SSH or locally.
- **Not an experiment tracker, workflow engine, or secrets manager.** Task meaning, metrics,
  dependencies, and result validation belong to your project.
- **No raw Slurm options, fractional GPUs, or cross-filesystem copies.**

See the [changelog](https://github.com/edoski/servatus/blob/main/CHANGELOG.md) for release notes.
