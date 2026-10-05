# Servatus

Durable Slurm sweeps from a laptop or a login node, with atomic publication of their results.

Servatus keeps one durable record of a sweep: which Tasks exist, which allocations were submitted,
and what Slurm accepted. It records its intent before each `sbatch`, so a dropped SSH connection
never silently doubles work. It tells you which Tasks are running, finished, failed, or uncertain,
and it never retries anything on its own. Inside the job, workers keep resumable checkpoints in a
private workspace and publish results atomically: a result is either absent or complete.

Servatus has no runtime dependencies and needs no Python on the cluster to submit work. It talks to
Slurm through plain OpenSSH, or runs the Slurm commands directly when you are on a login node.
Workers that publish results need `servatus` where they run, for example in the image.

```sh
pip install servatus      # Python 3.11+, Linux or macOS
```

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
checks that the profile loads and fits its ceilings, and that `sbatch --version` answers.

### 2. Write the worker

Each Task runs one process in its own `srun` step. The process gets its Task's arguments, stdin
bytes, and environment, plus `SERVATUS_TASK_KEY`, `SERVATUS_ALLOCATION_ID`, `SERVATUS_SLOT`,
`SERVATUS_JOB_ID`, and `SERVATUS_RESTART_COUNT`.

```python
# train.py, in work_root
import os, signal, sys, threading
from pathlib import Path

from mylab import Trainer  # your code
from servatus import Workspace
from servatus.errors import DestinationExists

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
requeue: a requeued or resubmitted Task reopens it and resumes. `publish` commits `results/<key>`
atomically and only then removes the private work.

### 3. Define the Tasks

Either a JSONL file (one Task per line; `stdin_file` and `env` are optional, and relative paths
resolve against the JSONL file's directory):

```json
{"key": "lr-1e-3", "args": ["python", "train.py"], "stdin_file": "configs/lr-1e-3.json"}
{"key": "lr-3e-4", "args": ["python", "train.py"], "stdin_file": "configs/lr-3e-4.json", "env": {"WANDB_MODE": "offline"}}
```

`servatus ensure state tasks.jsonl` creates the campaign, or appends the Tasks it has not seen.
Or use a Python launcher. Run it where it can see the results directory: on the login node (with
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
receipts = campaign.submit(plan).receipts
```

`ensure` is safe to rerun: existing Tasks must be identical by key (build them deterministically)
and new keys are appended in order. With a probe, Tasks whose results exist are never resubmitted;
a plan made with a probe must be submitted by a Campaign with one, so not from the CLI.

### 4. Plan, submit, watch

```sh
servatus plan state --output plan.json     # shows selected, held, deferred; saves the plan
servatus submit state plan.json
servatus status state
servatus logs state --task lr-1e-3 --output lr-1e-3.log
servatus plan state --retry-failed --output retry.json    # later: rerun failed or cancelled Tasks
```

Without `--output`, `plan` only prints. `submit` rebuilds the reviewed plan, refuses it with
`StalePlan` if the campaign changed, and pings Slurm and rechecks eligibility before recording
anything.

## Concepts

A **Campaign** is a directory holding an ordered roster of **Tasks** and every **Attempt** to run
them. A **Profile** pairs a **Target** (where and how to run, with ceilings that guard against
mistakes) with **Resources** (what each Task needs). A **Plan** packs eligible Tasks into
single-node **allocations** of concurrent steps. Before each `sbatch`, submission durably records an
**intent**; Slurm's job number becomes a **Receipt**. Acceptance is not completion: Servatus reads
`squeue` and `sacct`, and an optional **result probe** says which results are valid. Unselected
Tasks are **held** ([glossary](https://github.com/edoski/servatus/blob/main/docs/CONTEXT.md)).

| Hold | Meaning | What to do |
| --- | --- | --- |
| `VALID` | The probe reports a valid result. | Nothing. |
| `SUBMITTED` | The Task has accepted work and was not requested for retry. Slurm is not consulted. | `status`; to rerun, `--retry KEY` or `--retry-failed`. |
| `ACTIVE` | Retry requested, but Slurm still holds earlier work (queued, running, held, requeued). | Wait, or `cancel`. |
| `UNRESOLVED` | An intent has no recorded outcome. | See the recovery playbook. |
| `UNOBSERVABLE` | Slurm's evidence for earlier work is missing or `UNKNOWN`. | `--retry KEY --allow-duplicate-risk KEY`. |
| `NOT_REQUESTED` | Excluded by `--only`. | Nothing. |

`--retry-failed` selects Tasks whose latest accepted work failed or was cancelled (judged by the
Task's own step when known). Explicit `--retry` keys are strict: an unknown key, a valid result, a
Task never accepted, or `UNKNOWN` evidence without `--allow-duplicate-risk` refuses the whole plan
(`PlanRefused`, listing every key). Duplicate risk can only be acknowledged for explicit keys.
Eligible Tasks beyond the target's `max_allocations_per_submit` are **deferred** to the next plan.

## Recovery playbook

| Exit | Meaning | Next step |
| --- | --- | --- |
| 0 | Success. | |
| 1 | Error: bad input, refused or stale plan, a shape `validate` rejected, conflicting or corrupt state. | Read the message. `StalePlan`: plan again. |
| 2 | Usage error. | Check `servatus COMMAND --help`. |
| 3 | Submission interrupted. Some allocations may be unresolved. | Run `status`, then resolve them as below. |
| 75 | Cluster unavailable or campaign busy. | Retry later. A failed ping records nothing. |

An **unresolved allocation** has a durable intent but no receipt: the `sbatch` call may or may not
have reached Slurm. Servatus holds its Tasks until you resolve it.

1. `servatus reconcile state ALLOCATION_ID` records the receipt if Slurm holds exactly one job
   with that allocation's identity (an already accepted allocation just prints its receipt).
2. Otherwise check `squeue`/`sacct` yourself. Record a job you find with
   `servatus mark-accepted state ALLOCATION_ID JOB_ID [--cluster NAME]`, or run
   `servatus mark-not-submitted state ALLOCATION_ID` if you are sure it never reached Slurm.
3. Plan again. History is never erased; a new Attempt is added.

`servatus cancel state --task KEY` runs `scancel` for every accepted allocation of that Task not
already known to be finished (`--allocation ID` names one). A packed allocation stops all its
Tasks. Nothing is retried; once Slurm reports the work cancelled, `--retry-failed` selects it.

## Publication

Publication works with or without a Campaign. Work, sources, stages, and destinations must share
one filesystem; Servatus never copies.

**`publish(destination, build, *, retire=None, mode=None)`** builds a directory in a private stage
and commits it with a no-replace rename. The builder receives a `Draft`, writes into `draft.path`,
validates, and returns; using the draft afterwards raises `RuntimeError`.

```python
from servatus import publish


def build(draft):
    (draft.path / "summary.json").write_text('{"ok": true}\n')
    draft.link_tree("runs/42/plots", "plots")  # hard-link every regular file under a tree


publish("outputs/run-42", build, mode=0o755)
```

**`publish_file(destination, write, *, mode=None)`** publishes one regular file. The writer
receives `<private stage>/<destination name>`, not yet created, and may create it any way it likes
(even a temporary file renamed into place); anything else it leaves in the stage is discarded.

```python
import numpy as np

from servatus import publish_file

weights = np.zeros((4, 4))
publish_file("outputs/weights.npy", lambda path: np.save(path, weights))
```

**`Workspace(destination, *, identity)`** is private, resumable work bound to opaque identity bytes
(see the worker). `workspace.publish(build, *, mode=None)` publishes, then removes the private work;
`workspace.discard()` removes it unpublished. Entering raises `Busy` while another process holds
it, `WorkspaceConflict` (naming the path) for work bound to another identity, and, when the
destination exists, `DestinationExists` after removing this identity's leftover private work.

**Children.** `parent.child(name, identity=...)` gives concurrent workers their own resumable
results beneath one future destination; the parent publishes once you decide they are all valid:

```python
from servatus import Workspace

parent = Workspace("outputs/study", identity=b"study v1")
with parent.child("method-a", identity=b"method-a v1") as child:
    (child.path / "result.bin").write_bytes(b"...")  # create or resume the child's work
    child.publish(lambda draft: draft.link(child.path / "result.bin", "result.bin"))
with parent as workspace:
    workspace.publish(lambda draft: draft.link_tree(workspace.path / "method-a", "method-a"))
```

**Modes.** Stages stay owner-only (0700) until commit. Just before commit, `publish` sets every
directory in the tree to `mode` (default `0o777 & ~umask`) and leaves file modes as written;
`publish_file` sets the file to `mode` (default `0o666 & ~umask`), overriding the writer's choice.

**Hard links alias.** `Draft.link` and `link_tree` publish the *same inode* as their source, so
rewrite sources only by replacement (a new file and `os.replace`), never in place.

`retire=` removes one owner-only sibling directory after the destination is durably committed.
Unprovable cleanup never fails a publication: it sets `cleanup_pending` and warns. An existing
destination raises `DestinationExists`; invalid input (an unsafe draft path, a missing parent, a
non-sibling `retire`, a bad `mode`) raises `ConfigurationError`. Linux without `renameat2` uses a
documented fallback ([SECURITY.md](https://github.com/edoski/servatus/blob/main/SECURITY.md)).

## Configuration reference

`Profile.load(path, *, name=None)` reads `SERVATUS.toml`. An explicit name wins over
`default_profile`; a sole profile selects itself. Unknown keys are rejected everywhere. Durations
are `[D-]H:MM:SS`; time limits round up to whole minutes. An allocation of `n` Tasks requests `n`
times the per-Task CPUs, memory, and GPUs, for the per-Task time limit.

| `[target]` key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `host` | string | omitted | SSH destination (`[user@]host` or ssh_config alias). Omitted: run Slurm locally. |
| `slurm_bin` | absolute path | required | Directory holding `sbatch`, `squeue`, `sacct`, `scancel`, `srun`. |
| `work_root` | absolute path | required | Working directory of every Task step. |
| `log_root` | absolute path | required | Slurm log directory (must exist). |
| `partitions` | array of strings | required | Partitions the job may use. |
| `max_tasks_per_allocation`, `max_cpus_per_allocation`, `max_memory_mib_per_allocation` | int | required | Ceilings per allocation. |
| `max_time_limit` | duration | required | Wall-time ceiling. |
| `account`, `qos`, `constraint` | string | omitted | Passed to `sbatch` when set. |
| `gpu_gres`, `max_gpus_per_allocation` | string, int | omitted, 0 | Count-free GPU GRES (`gpu:a100`) and its ceiling; set both or neither. |
| `max_allocations_per_submit` | int | no cap | Allocations per plan; the rest are deferred. |
| `max_script_bytes` | int | 1048576 | Batch script size bound. |
| `apptainer`, `image` | absolute path | omitted | Apptainer launcher: set both, or neither for the direct launcher. |
| `binds` | array of strings | `[]` | Apptainer only: `SRC[:DST[:ro\|rw]]` mounts; `work_root` is always bound. |

| `[resources]` key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `cpus`, `memory_mib` | int | required | CPUs and MiB per Task. |
| `time_limit` | duration | required | Wall time of the allocation. |
| `gpus` | int | 0 | Whole GPUs per Task. |
| `signal_before_end` | duration | omitted | Send SIGUSR1 to each Task this long before the limit. |

In Python these are `Target`, `Apptainer`, `Resources`, and `Profile`;
`servatus.campaign.capacity(target, resources)` returns how many Tasks fit one allocation.

**Launchers.** With Apptainer, each step runs `apptainer run --cleanenv` on the image (its
runscript receives the Task's `args`) from `work_root`, with `work_root` plus `binds` mounted and
the Task's `env` passed as `APPTAINERENV_*`; GPU steps get `--nv` and Slurm's step-local
`CUDA_VISIBLE_DEVICES`. Without Apptainer, each step runs the Task's absolute `args[0]` under
`env -i` with the same variables. Either way, Task stdin and environment are written into the batch
script and are visible to cluster administrators: never put secrets in Tasks.

**SSH.** Servatus runs `ssh -T -o BatchMode=yes -o LogLevel=ERROR HOST`, so it never prompts, and
ignores login banners and shell-startup output. Give the host `ControlMaster auto`,
`ControlPath ~/.ssh/control-%C`, and `ControlPersist 30m` in `~/.ssh/config`: a persistent master
connection makes calls fast and, with multi-factor login, lets you authenticate once
(`ssh hpc true`) first. Keys must load without a passphrase prompt, for example via `ssh-agent`.

## CLI reference

`STATE` is the campaign directory. `plan` and `doctor` read `./SERVATUS.toml` unless
`--config PATH` is given. Output is human-readable; every command except `logs` accepts `--json`.
`status --json` prints a `servatus.status/1` document (`tasks`, `attempts`, `counts`, `quiescent`,
`results_ready`); saved plans are `servatus.plan/1` documents.

| Command | Purpose |
| --- | --- |
| `create STATE TASKS.jsonl [--appendable]` | Create a campaign; fails if it exists. Sealed unless `--appendable`. |
| `ensure STATE TASKS.jsonl [--sealed]` | Create (appendable unless `--sealed`), or append unseen Tasks; existing keys must match. |
| `append STATE TASKS.jsonl` | Append new Tasks to an appendable campaign. |
| `seal STATE` | End authoring irreversibly. |
| `plan STATE [--profile NAME] [--output FILE]` | Show selected, held, and deferred Tasks; `--output` saves the plan (0600, never overwrites). |
| `  --retry KEY`, `--retry-failed` | Retry named Tasks, or every Task whose latest work failed or was cancelled. |
| `  --allow-duplicate-risk KEY` | Retry a named Task whose earlier work has `UNKNOWN` evidence. |
| `  --only KEY`, `--tasks-per-allocation N`, `--show-scripts` | Restrict the plan; lower packing; print batch scripts (sensitive). |
| `validate STATE PLAN` | `sbatch --test-only` once per distinct allocation shape; exit 1 if any is rejected. |
| `submit STATE PLAN` | Submit the reviewed plan. Exit 3 if interrupted. |
| `status STATE [--offline]` | Task and allocation status; `--offline` skips Slurm. |
| `logs STATE [--task KEY] [--allocation ID] [--bytes N] [--output FILE]` | Bounded log tail; writes 0600; refuses a terminal. |
| `reconcile STATE ALLOCATION` | Resolve an unresolved allocation from Slurm evidence. |
| `mark-accepted STATE ALLOCATION JOB_ID [--cluster NAME]` | Record a job you found yourself. |
| `mark-not-submitted STATE ALLOCATION` | Record that an allocation never reached Slurm. |
| `cancel STATE [--task KEY]... [--allocation ID]...` | `scancel` matching accepted allocations not known to be finished. |
| `doctor [--profile NAME]` | Check the profile and the scheduler connection. |

`logs --task KEY` reads that Task's step log in its latest accepted allocation, `--allocation ID`
the allocation log, and both that Task's step log in that allocation; `--bytes` is 1 to 1048576
(default 65536). Key and allocation options repeat. `servatus --version` prints the version.

## Guarantees

- Intent is durable before each `sbatch`. An uncertain submission blocks its Tasks until it is
  resolved; nothing is retried without an explicit decision.
- Submission executes exactly the reviewed plan, or refuses it as stale or tampered.
- Campaign state is owner-only, validated on every read, atomically replaced, and synced.
- Scheduler evidence is bounded, identity-checked, and read from each Attempt's original target.
- A published destination is absent or complete and is never overwritten. Content is synced before
  commit and the parent after it.
- Private work survives failure and is removed only after a durable publication.
- Application meaning stays with you: Servatus never sees schemas or decides completion.

The [architecture decisions](https://github.com/edoski/servatus/blob/main/docs/adr/README.md)
explain each guarantee and the current scope, and
[SECURITY.md](https://github.com/edoski/servatus/blob/main/SECURITY.md) the trust model. To test
your launchers, pass `servatus.testing.FakeScheduler()`, an in-memory Slurm, as `connect=` to
`Campaign` or `servatus.cli.main`.

## Non-goals

- **No job arrays or multi-node jobs.** One Task may use several GPUs on its node (`torchrun`).
- **No automatic retry, background polling, or scheduler plugins.** One native Slurm lane.
- **Not an experiment tracker, workflow engine, or secrets manager.** Task meaning belongs to you.
- **No raw Slurm options, fractional GPUs, or cross-filesystem copies.**

See the [changelog](https://github.com/edoski/servatus/blob/main/CHANGELOG.md) for release notes.
