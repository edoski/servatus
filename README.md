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
checks that the profile fits its ceilings and that the cluster has what Servatus needs. See
[Clusters](#clusters) for complete configurations and the steps to onboard a new user.

### 2. Write the worker

Each Task runs one process in its own `srun` step. The process gets its Task's arguments, stdin
bytes, and environment, plus these reserved variables (no `SLURM_*` variable reaches it):

| Variable | Meaning |
| --- | --- |
| `SERVATUS_TASK_KEY` | The Task's key. |
| `SERVATUS_ALLOCATION_ID`, `SERVATUS_SLOT` | The allocation running it and the Task's 0-based slot in it. |
| `SERVATUS_JOB_ID` | Slurm's job number. |
| `SERVATUS_JOB_END_TIME` | When Slurm will end the job, in UNIX seconds: its projected end when the job started (`SLURM_JOB_END_TIME`). A later `scontrol update TimeLimit` does not change it. |

A chain step also gets the [continuation variables](#the-application-protocol).

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

The `Workspace` directory is bound to the identity bytes and survives failure, preemption, and time
limits: a resubmitted Task reopens it and resumes. Slurm never requeues Servatus jobs on its own
(`--no-requeue`); a rerun is always a new, explicit plan. `publish` commits `results/<key>`
atomically and only then removes the private work.

### 3. Define the Tasks

Either a JSONL file (one Task per `\n`-terminated line; `stdin_file` and `env` are optional, and
relative paths resolve against the JSONL file's directory):

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
decision = plan.decision  # what you review: selected, held, and deferred Tasks
print(f"{len(decision.selected)} selected, {len(decision.held)} held")
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
servatus plan state --retry-failed --output retry.json    # later: rerun failed or stopped Tasks
```

Without `--output`, `plan` only prints. `submit` rebuilds the reviewed plan, refuses it with
`StalePlan` if the campaign changed, and pings Slurm and rechecks eligibility before recording
anything. `status` shows each Task's execution: `QUEUED`, `RUNNING`, `ENDING` (Slurm
says the job ended; the outcome is final a few minutes later, once Slurm forgets it),
`SUCCEEDED` or `FAILED` with its step's exit code, `STOPPED` (cancelled, out of time, never
started, or the step exited 75 to say "stopped cleanly, continue me"), or `LOST` (the job vanished
without recording how it ended, for example on a node failure). A Task without an observed
execution shows `NOT_SUBMITTED`, `UNRESOLVED` (reconcile it), or `ACCEPTED` (with `--offline`).
`RUN TIME` is how long the Task has run, summed over every allocation that ran it; gaps between
allocations do not count, and a `~` marks a total that is still growing or that counts a `LOST` run
at its whole time limit.

Servatus needs no Slurm accounting (`sacct`). It reads `squeue` for work Slurm still lists, and
the run markers each batch script writes to `log_root`, by name and mtime: a start marker before
any step, one end marker per step as soon as it returns, and a stop marker when the job is
interrupted. A step's end marker shows as `SUCCEEDED` or `FAILED` at once. Run time is the span
between those mtimes, which the file server stamps, so it never mixes clocks. A finished job stays in `squeue` for the
cluster's `MinJobAge` (typically 5 minutes), and a missing marker means something only after that,
when every marker is visible on the login node. Until then a cancelled Task shows `ENDING`
and retries are held `ACTIVE`, so `--retry-failed` picks failed work up a few minutes after it ends.
This needs a nonzero `MinJobAge` well above how long the login node caches directory listings (NFS:
about a minute); `scontrol show config | grep MinJobAge` shows it.

## Concepts

A **Campaign** is a directory holding an ordered roster of **Tasks** and every **Attempt** to run
them. A **Profile** pairs a **Target** (where and how to run, with ceilings that guard against
mistakes) with **Resources** (what each Task needs). A **Plan** packs eligible Tasks into
single-node **allocations** of concurrent steps. Before each `sbatch`, submission durably records an
**intent**; Slurm's job number becomes a **Receipt**. Acceptance is not completion: Servatus reads
`squeue` and the run markers in `log_root`, and an optional **result probe** says which results
are valid. Unselected Tasks are **held**
([glossary](https://github.com/edoski/servatus/blob/main/docs/CONTEXT.md)).

| Hold | Meaning | What to do |
| --- | --- | --- |
| `VALID` | The probe reports a valid result. | Nothing. |
| `SUBMITTED` | The Task has accepted work and was not requested for retry. Slurm is not consulted. | `status`; to rerun, `--retry KEY` or `--retry-failed`. |
| `ACTIVE` | Slurm still lists earlier work of the Task: queued, running, or finished but not yet purged. | Wait a few minutes, or `cancel`. |
| `UNRESOLVED` | An intent has no recorded outcome. | `servatus reconcile STATE`. |
| `SPENT` | Under `--budget`, the Task's run time leaves no room for one more link. | Nothing, or extend it with a larger budget. |
| `NOT_REQUESTED` | Excluded by `--only`. | Nothing. |

`--retry-failed` selects Tasks whose current work is `FAILED` or `STOPPED`. It never selects
`LOST` work: the job vanished without recording an outcome, and during a network partition its
step may even still run. Rerun a `LOST` Task by naming it, `--retry KEY`; naming it is your
acknowledgement. Explicit `--retry` keys are strict: an unknown key, a valid result, a Task never
accepted, or a Task whose earlier work Slurm still lists refuses the whole plan (`PlanRefused`,
listing every key). Eligible Tasks beyond the target's `max_allocations_per_submit` are
**deferred** to the next plan.

## Budgeted continuation

A long training Task can resume from its own checkpoint. Give its plan a lifetime run-time budget,
and Servatus queues enough allocations, called **links**, to spend it. Each link continues where
the previous one stopped, and nothing needs to run on your machine while the chain progresses.

```toml
[profiles.l40s.resources]
time_limit = "16:00:00"
signal_before_end = "00:10:00"   # required for continuation
```

```sh
servatus plan state --profile l40s --budget 3-00:00:00 --output plan.json
#   continuation: lifetime budget 3-00:00:00 per Task
#     az-small  run time 00:00:00, budget left 3-00:00:00, 5 links
#   selected: 1 Task in 5 allocations
#     9a1c5d0e7b3f2a8c4d6e1b09  1 Task, 32 CPUs, 64000 MiB, 1 GPUs, 16:00:00, link 1/5
#     ...                       1 Task, 32 CPUs, 64000 MiB, 1 GPUs, 08:00:00, link 5/5
servatus submit state plan.json   # queues all five links now; close the laptop
servatus status state             # RUN TIME 1-09:12:03 / 3-00:00:00, LINK 3/5
```

The budget is a lifetime total per Task: full `time_limit` links, then one shorter link with the
rest rounded down to whole minutes, so a Task's links never add up to more than its budget. A rest
within a minute of `signal_before_end` is dropped with a warning. Tasks packed together with
different run times share a chain sized for the neediest; the plan warns, and each Task still
learns its own budget. `--max-links K` caps a chain at K links (1 to 32;
alone, it plans K full links). Each link starts only after every earlier link of its chain has
ended, and Slurm deletes the rest of a chain, before it allocates anything, once a link hands
nothing on.

### The application protocol

At `signal_before_end` before each link's limit, every step gets `SIGUSR1`.

| Your step | Status | Chain |
| --- | --- | --- |
| exits 0 | `SUCCEEDED` | ends for this Task |
| exits 75 at any time | `STOPPED (yielded)`, exit 75 | continues |
| gets `SIGUSR1`, then exits nonzero or is stopped at the limit | `STOPPED (time limit)` | continues |
| exits nonzero before `SIGUSR1` | `FAILED` | ends; `--retry-failed` reruns it |
| is cancelled or preempted | `STOPPED` | ends |
| exits 75 on its last planned link | `STOPPED (budget spent)` | ends |

So an application that never handles `SIGUSR1` still continues: it dies of the signal (exit 138),
which counts as stopped at the limit. The chain's last link gets the signal as a plain allocation
does, so there a nonzero exit after `SIGUSR1` reads `FAILED` and a stop at the limit reads
`STOPPED`. On the last link, exit 0 when done, or 75 to read `budget spent`. Every chain step also
receives these reserved variables:

| Variable | Meaning |
| --- | --- |
| `SERVATUS_BUDGET` | The Task's lifetime run-time budget in seconds. Unset without `--budget`. |
| `SERVATUS_RUN_TIME` | Seconds the Task ran before this allocation, across every allocation it ever had, measured. |
| `SERVATUS_LINK`, `SERVATUS_LINKS` | This link's 1-based position and the Task's planned number of links. `LINK == LINKS` means this is its last link. |

An application that counts its own budget takes `SERVATUS_BUDGET` as its total, and
`SERVATUS_RUN_TIME` plus its own elapsed time (or its own checkpointed counter) as what it used. It
exits 0 once the budget is spent, and a resumed run whose budget is already spent exits 0 at once:

```python
import os, signal, sys, time

started = time.monotonic()
budget = int(os.environ.get("SERVATUS_BUDGET", config.budget_seconds))
used_before = int(os.environ.get("SERVATUS_RUN_TIME", 0))
last = os.environ.get("SERVATUS_LINK") == os.environ.get("SERVATUS_LINKS")


def on_usr1(*_):
    save(state)
    spent = used_before + time.monotonic() - started >= budget
    sys.exit(0 if spent or last else 75)


signal.signal(signal.SIGUSR1, on_usr1)
```

### Extending a Task

A budget is a total, not an increment, so planning the same budget again selects nothing: the Task
is held `SPENT`. To give a finished Task more time, name it with a larger budget:

```sh
servatus plan state --retry az-small --budget 1-12:00:00 --output extend.json
#     az-small  run time 1-00:00:41, budget left 11:59:19, 1 link, extends SUCCEEDED
```

`--retry KEY --budget D` is the only way to extend a Task. It accepts a `SUCCEEDED` Task, and one
whose probe reports `VALID`, when its run time leaves room under `D`; the plan names each as an
extension. Without `--budget`, a `VALID` Task is refused as before. `--retry-failed --budget D`
selects `FAILED` and `STOPPED` Tasks with room left and holds the rest `SPENT`. The step resumes
from its own checkpoint; Servatus never touches its outputs, so an application that publishes an
immutable destination at the old budget must publish the extended result elsewhere.

### Cancelling a chain

`servatus cancel state --task KEY` cancels every queued or running link of the Task, newest first,
so no later link becomes eligible in between. `--allocation ID` on a link cancels it and every
later link of its chain. Cancel through Servatus: a raw `scancel` of a running link after its
pre-limit signal reads as a time limit, and the chain continues.

### What changes in the CLI and JSON

| Surface | Change |
| --- | --- |
| `plan --budget DURATION` | Lifetime run-time budget per selected Task (`[D-]HH:MM:SS`, rounded up to whole minutes). Needs `signal_before_end`. |
| `plan --max-links K` | At most K links per chain, 1 to 32. Without `--budget`, K full links. |
| `plan` output | A `continuation` line; per Task its run time, budget left, link count, and `extends SUCCEEDED` or `extends VALID`; each allocation's `link j/K`. |
| `plan` holds | `SPENT`: the Task's run time leaves no room under the budget. |
| `status` | `RUN TIME` (`/ budget` in a budgeted chain) and `LINK` (`3/5`, `-` for one allocation); details `time limit`, `yielded`, `budget spent`, `never started`. |
| `status --json` | `servatus.status/3`: each Task's `run_time` and `link`, each Attempt's `link` and timed `observation`. |
| Saved plans | `servatus.plan/3`: the Decision gains `continuation`, `spent` (each selected Task's measured run time), and `extensions`. |
| Library | `Campaign.plan(profile, continuation=Continuation(budget="3-00:00:00"))`; `Plan.extensions`; `Link`, `RunTime`, `StopReason`, `Extension`, `YIELD_EXIT`, `MAX_LINKS`. |
| Exit codes | Unchanged. |

## Recovery playbook

| Exit | Meaning | Next step |
| --- | --- | --- |
| 0 | Success. | |
| 1 | Error: bad input, refused or stale plan, a shape `validate` rejected, conflicting or corrupt state, or an unexpected error (one line, never a traceback). | Read the message. `StalePlan`: plan again. |
| 2 | Usage error. | Check `servatus COMMAND --help`. |
| 3 | Submission interrupted. Some allocations may be unresolved. | Run `status`, then resolve them as below. |
| 75 | Cluster unavailable, campaign busy, a transient filesystem failure (no space, quota, I/O), or `reconcile` still waiting for evidence. | Retry later. A failed ping records nothing. |
| 130 | Interrupted (Ctrl-C). Recorded intent survives. | Run `status`. |

An **unresolved allocation** has a durable intent but no receipt: the `sbatch` call may or may not
have reached Slurm. Servatus holds its Tasks until you resolve it.

1. `servatus reconcile state` resolves every unresolved allocation from Slurm's evidence. A job
   that `squeue` lists with the allocation's identity, or else one named by its run markers, is
   recorded as accepted. With neither, an intent at least 15 minutes old is recorded as not
   submitted; a younger one waits, and `reconcile` prints when to run it again and exits 75.
   Rerunning it is always safe.
2. Plan again. History is never erased; a new Attempt is added.

If Slurm later lists a job for an allocation recorded as not submitted (an `sbatch` that arrived
very late), `status` warns, names the allocation and job, and holds its Tasks `ACTIVE` until the job
leaves the queue, so nothing new is submitted beside it. `servatus cancel state --allocation ID`
stops it.

`servatus cancel state --task KEY` runs `scancel --me --name=servatus-<allocation id>` for every
allocation of that Task that Slurm still lists as queued or running (`--allocation ID` names one);
a packed allocation stops all its Tasks. Cancelling by name can never hit an unrelated job that
reused a job number, and a job that already ended is not an error. Every allocation is tried;
failures exit 75, naming what was and was not cancelled. Nothing is retried; once Slurm purges the
cancelled job, its Tasks are `STOPPED` and `--retry-failed` selects them.

## Publication

Publication works with or without a Campaign. Work, sources, stages, and destinations must share
one filesystem; Servatus never copies.

**`publish(destination, build, *, mode=None)`** builds a directory in a private stage
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
it, `WorkspaceConflict` (naming the path) for work bound to another identity or holding files but
no identity, and, when the destination exists, `DestinationExists` after removing this identity's
leftover private work. Removal deletes `work` before the identity record, so an interrupted
removal is finished by the next entry with the same identity.

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

Entering a finished child raises `DestinationExists` for that child only: the parent has not
necessarily published, so do not read it as "the whole result exists".

**Modes.** Stages stay owner-only (0700) until commit. Just before commit, `publish` sets every
directory in the tree to `mode` (default `0o777 & ~umask`) and leaves file modes as written;
`publish_file` sets the file to `mode` (default `0o666 & ~umask`), overriding the writer's choice.

**Hard links alias.** `Draft.link` and `link_tree` publish the *same inode* as their source, so
rewrite sources only by replacement (a new file and `os.replace`), never in place.

Unprovable cleanup never fails a publication: it sets `cleanup_pending` and warns. Where a
directory cannot be `flock`ed (NFS), Servatus locks an owner-only `.servatus.lock` file in it
instead and leaves it there; another user's lock file is refused. Linux without `renameat2` uses a
documented fallback ([SECURITY.md](https://github.com/edoski/servatus/blob/main/SECURITY.md)).

**Errors.** Only an existing destination raises `DestinationExists`. Bad input or an unusable
location raises `ConfigurationError`: an unsafe draft path; a link onto an occupied draft path; a
missing, unwritable, or overlong path; a missing link source; paths on different filesystems; a
platform or filesystem without an atomic no-replace rename or a working lock; a writer that creates
nothing; a `link_tree` source containing the draft; a file its owner cannot read; a bad `mode`. No
space, quota, I/O errors, or too many open files raise `Unavailable`; `UnsafeFilesystem` means an
entry was substituted or has the wrong type or owner. No raw `OSError` escapes, but exceptions from
your builder or writer propagate unchanged.

## Configuration reference

`Profile.load(path, *, name=None)` reads `SERVATUS.toml`. An explicit name wins over
`default_profile`; a sole profile selects itself. Unknown keys are rejected everywhere. Durations
are `[D-]H:MM:SS`; time limits round up to whole minutes. An allocation of `n` Tasks requests `n`
times the per-Task CPUs, memory, and GPUs, for the per-Task time limit.

| `[target]` key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `host` | string | omitted | SSH destination (`[user@]host` or ssh_config alias). Omitted: run Slurm locally. |
| `slurm_bin` | absolute path | required | Directory holding `sbatch`, `squeue`, `scancel`, `srun`. |
| `work_root` | absolute path | required | Working directory of every Task step; no `,` or `:`. |
| `log_root` | absolute path | required | Slurm logs and run markers (must exist, shared with the login node); no `%`. |
| `partitions` | array of strings | required | Partitions the job may use. |
| `max_tasks_per_allocation`, `max_cpus_per_allocation`, `max_memory_mib_per_allocation` | int | required | Ceilings per allocation. |
| `max_time_limit` | duration | required | Wall-time ceiling. |
| `account`, `qos` | string | omitted | Passed to `sbatch` when set. |
| `constraint` | string | omitted | Slurm feature expression (`a100\|h100`, `intel&ib`), passed to `sbatch` as one argument. |
| `gpu_gres`, `max_gpus_per_allocation` | string, int | omitted, 0 | Count-free GRES (`gpu`, `gpu:a100_80GB`, `shard`) and its ceiling; set both or neither. `gpus` counts units of it, and Slurm must set each step's `CUDA_VISIBLE_DEVICES`. |
| `max_allocations_per_submit` | int | no cap | Allocations per plan; the rest are deferred. |
| `max_script_bytes` | int | 1048576 | Batch script size bound. |
| `apptainer`, `image` | absolute path | omitted | Apptainer launcher: set both, or neither for the direct launcher. `image` has no `:`. |
| `binds` | array of strings | `[]` | Apptainer only: `SRC[:DST[:ro\|rw]]` mounts; `work_root` is always bound. |

| `[resources]` key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `cpus`, `memory_mib` | int | required | CPUs and MiB per Task. |
| `time_limit` | duration | required | Wall time of the allocation. |
| `gpus` | int | 0 | Units of `gpu_gres` per Task: whole GPUs, or whole shards with `shard`. |
| `signal_before_end` | duration | omitted | Send SIGUSR1 to each Task this long before the limit. |

In Python these are `Target`, `Apptainer`, `Resources`, and `Profile`;
`servatus.campaign.capacity(target, resources)` returns how many Tasks fit one allocation, and
`servatus.campaign.diagnose(profile)` returns what `servatus doctor` prints as a `Diagnosis`.

**Launchers.** With Apptainer, each step runs `apptainer exec --cleanenv IMAGE ARGS...` from
`work_root`, so the Task's `args` run directly and the image's runscript is ignored (a relative
`args[0]` resolves on the container's `PATH`; empty `args` fail at planning). `work_root` plus
`binds` are mounted, and the Task's `env` reaches the container as `APPTAINERENV_*` assignments
on `srun`, never on a command line. GPU steps get `--nv` and Slurm's step-local
`CUDA_VISIBLE_DEVICES`. The batch script sets `PATH=/usr/bin:/bin`, which `--nv` needs to find
the host's `nvidia-smi`, MPS daemon and `libcuda`. Without Apptainer, each step runs the Task's
absolute `args[0]` under `env -i` with the same variables passed as arguments. Either way, Task
stdin and environment are written into the batch script, which cluster administrators can read,
and a direct step's environment also shows in its command line to other users on the node. Keep
secrets out of Tasks.

**SSH.** Servatus runs `ssh -T -o BatchMode=yes -o LogLevel=ERROR -o ConnectTimeout=8 -o
ConnectionAttempts=3 HOST`, so it never prompts, ignores login banners and shell-startup output,
and retries a connection that never opens. A login that stalls after connecting is not retried, and
commands themselves never are; each has 30 s to finish, connecting included. With `ProxyJump`, set
those two options on the jump host too. From your environment it passes only `HOME`, `LOGNAME`,
`USER`, `SSH_AUTH_SOCK`, `KRB5CCNAME`, and `KRB5_CONFIG`, so agent and Kerberos logins work. Give
the host `ControlMaster auto`, `ControlPath ~/.ssh/control-%C`, `ControlPersist 30m`,
`ServerAliveInterval 10`, and `ServerAliveCountMax 2` in `~/.ssh/config`: a persistent master
connection makes calls fast and, with multi-factor login, lets you authenticate once
(`ssh hpc true`) first, and the keepalive ends a master whose network went away. Keys must load
without a passphrase prompt, for example via `ssh-agent`.

## Clusters

A new cluster is a new profile, not new code. Each configuration below passed `servatus doctor` for
every profile, with `NAME.SURNAME` and the paths filled in. Both clusters run Slurm with `MinJobAge`
of 300 s and `proctrack/cgroup`, and neither needs `sacct`. Apptainer exists only on their compute
nodes, at `/usr/bin/apptainer`, so `doctor` cannot check it from the login node.

### Student cluster (`giano`)

Slurm 24.11. Partition `rtx2080` has 8 nodes of 4 CPUs, 42 GiB, and one RTX 2080 Ti. Partition `l40`
has 4 nodes of 8 CPUs, 64 GiB, and one L40. Home has a 400 MB quota, so everything lives in
`/scratch.hpc/NAME.SURNAME`, where files unread for 40 days are deleted. Publish what you want to
keep somewhere else before then.

```toml
default_profile = "rtx2080"

[target]
host = "university"
slurm_bin = "/usr/bin"
work_root = "/scratch.hpc/NAME.SURNAME/sweep"
log_root = "/scratch.hpc/NAME.SURNAME/sweep/logs"
apptainer = "/usr/bin/apptainer"
image = "/scratch.hpc/NAME.SURNAME/images/train.sif"
max_tasks_per_allocation = 1
max_time_limit = "3-00:00:00"

[profiles.rtx2080.target]
partitions = ["rtx2080"]
gpu_gres = "gpu"
max_gpus_per_allocation = 1
max_cpus_per_allocation = 4
max_memory_mib_per_allocation = 40960

[profiles.rtx2080.resources]
cpus = 4
memory_mib = 40960
gpus = 1
time_limit = "12:00:00"

[profiles.l40.target]
partitions = ["l40"]
gpu_gres = "gpu"
max_gpus_per_allocation = 1
max_cpus_per_allocation = 8
max_memory_mib_per_allocation = 61440

[profiles.l40.resources]
cpus = 8
memory_mib = 61440
gpus = 1
time_limit = "12:00:00"

# CPU-only work on the same nodes. Two Tasks share a node.
[profiles.cpu.target]
partitions = ["rtx2080"]
max_tasks_per_allocation = 2
max_cpus_per_allocation = 4
max_memory_mib_per_allocation = 32768

[profiles.cpu.resources]
cpus = 2
memory_mib = 16384
time_limit = "04:00:00"
```

Onboarding a student, in order:

1. **Add the SSH alias.** A login takes about 0.6 s and Servatus opens one connection per Slurm
   command, so share one. Add this to `~/.ssh/config`, then make a key if you have none
   (`ssh-keygen -t ed25519`) and run `ssh-copy-id university`:

   ```ssh-config
   Host university
       HostName giano.cs.unibo.it
       User NAME.SURNAME
       ControlMaster auto
       ControlPath ~/.ssh/control-%C
       ControlPersist 30m
       ServerAliveInterval 10
       ServerAliveCountMax 2
   ```

2. **Create the directories on scratch**, never in home:

   ```sh
   ssh university 'umask 077 && mkdir -p /scratch.hpc/$USER/sweep/logs /scratch.hpc/$USER/sweep/results /scratch.hpc/$USER/images'
   ```

3. **Build the image on a compute node.** The login node has no Apptainer, and the nodes have
   network access. Copy a definition like this to `/scratch.hpc/NAME.SURNAME/images/train.def`:

   ```text
   Bootstrap: docker
   From: python:3.12-slim

   %post
       pip install --no-cache-dir servatus
   ```

   Then build it, keeping Apptainer's cache out of home:

   ```sh
   ssh university srun --partition=rtx2080 --cpus-per-task=2 --mem=4G --time=00:20:00 \
     --chdir=/scratch.hpc/NAME.SURNAME/images \
     /usr/bin/env APPTAINER_CACHEDIR=/scratch.hpc/NAME.SURNAME/images/cache \
     /usr/bin/apptainer build train.sif train.def
   ```

   To run a virtual environment instead, make it under `work_root` with `pip3 install
   --no-cache-dir`, drop `apptainer` and `image` from `[target]`, and start each Task's `args` with
   the absolute path of its Python.

4. **Save the configuration above as `SERVATUS.toml`** next to your launcher, with your
   `NAME.SURNAME`, then check it:

   ```sh
   servatus doctor --profile rtx2080
   ```

5. **Continue with the [quickstart](#quickstart-a-sweep-in-10-minutes)** from step 2. `doctor`
   cannot see Apptainer or the image from the login node, so a first small Task proves those.

### Research cluster (`slogin`)

Slurm 23.11, for DISI staff and PhD students. The onboarding steps are the same with the alias
`research` (`HostName slogin.disi.unibo.it`), the partition `sbuild` for the image, and a profile
from this file for `doctor`. These profiles leave CPUs and memory for other users on shared nodes.

```toml
default_profile = "a100"

[target]
host = "research"
slurm_bin = "/usr/bin"
apptainer = "/usr/bin/apptainer"
image = "/scratch.hpc/NAME.SURNAME/images/train.sif"
work_root = "/scratch.hpc/NAME.SURNAME/sweep"
log_root = "/scratch.hpc/NAME.SURNAME/sweep/logs"
account = "default"
qos = "limitjob3node2"
max_tasks_per_allocation = 1
max_cpus_per_allocation = 128
max_memory_mib_per_allocation = 262144
max_gpus_per_allocation = 1
max_time_limit = "72:00:00"

[resources]
cpus = 64
memory_mib = 131072
gpus = 1
time_limit = "72:00:00"

[profiles.a100.target]
partitions = ["a100"]
gpu_gres = "gpu:a100_80GB"

# Two one-GPU runs per node, each on 104 CPUs: the 16 left over keep other users' small jobs from
# holding the node out of reach.
[profiles.l40s.target]
partitions = ["l40s"]
gpu_gres = "gpu:l40s"
max_tasks_per_allocation = 2
max_cpus_per_allocation = 208
max_memory_mib_per_allocation = 393216
max_gpus_per_allocation = 2

[profiles.l40s.resources]
cpus = 104
memory_mib = 196608

[profiles.l40.target]
partitions = ["l40"]
gpu_gres = "gpu:l40"
max_tasks_per_allocation = 2
max_gpus_per_allocation = 2
max_memory_mib_per_allocation = 196608

[profiles.l40.resources]
memory_mib = 98304

[profiles.h100.target]
partitions = ["h100pcie"]
gpu_gres = "gpu:h100pcie"

[profiles.cpu.target]
partitions = ["amdepyc9754"]
max_gpus_per_allocation = 0

[profiles.cpu.resources]
gpus = 0
cpus = 128
memory_mib = 262144

# CPU-only evaluations on the EPYC nodes, or the a100 node's idle CPUs when those are full.
[profiles.eval.target]
partitions = ["amdepyc9754", "a100"]
max_gpus_per_allocation = 0
max_time_limit = "06:00:00"
max_tasks_per_allocation = 8
max_cpus_per_allocation = 144

[profiles.eval.resources]
gpus = 0
cpus = 18
memory_mib = 32768
time_limit = "06:00:00"

[profiles.smoke.target]
partitions = ["l40"]
gpu_gres = "gpu:l40"
max_time_limit = "00:30:00"

[profiles.smoke.resources]
cpus = 16
memory_mib = 32768
time_limit = "00:30:00"

# One training task with most of the l40 node's CPUs, leaving two GPUs for other users.
[profiles.l40-solo.target]
partitions = ["l40"]
gpu_gres = "gpu:l40"
max_cpus_per_allocation = 112

[profiles.l40-solo.resources]
cpus = 112
memory_mib = 98304
```

## CLI reference

`STATE` is the campaign directory. `plan` and `doctor` read `./SERVATUS.toml` unless
`--config PATH` is given. Output is human-readable, with every non-printable character escaped
(except `--show-scripts`); every command except `logs` accepts `--json` for exact machine output.
`status --json` prints a `servatus.status/3` document (`tasks`, `attempts`, `counts`, `quiescent`,
`results_ready`, `warnings`). Each Task carries `run_time` (`{"total", "complete"}`, or `null`
offline), `detail` (why a `STOPPED` Task stopped: `never started`, `time limit`, `yielded`, or
`budget spent`), and `link`; each Attempt's `observation` carries the mtimes of its start, stop,
and limit markers, each slot's end (`{"exit_code", "at"}` or `null`), and which slots it was
handed. Saved plans are `servatus.plan/3` documents. Receipts, submit and
reconcile results, shape checks, and planned allocations are encoded with
`servatus.campaign.to_document`; a receipt is `{"allocation_id", "job", "task_keys"}` with an
integer job number.

| Command | Purpose |
| --- | --- |
| `create STATE TASKS.jsonl [--appendable]` | Create a campaign; fails if it exists. Sealed unless `--appendable`. |
| `ensure STATE TASKS.jsonl [--sealed]` | Create (appendable unless `--sealed`), or append unseen Tasks; existing keys must match. |
| `append STATE TASKS.jsonl` | Append new Tasks to an appendable campaign. |
| `seal STATE` | End authoring irreversibly. |
| `plan STATE [--profile NAME] [--output FILE]` | Show selected, held, and deferred Tasks; `--output` saves the plan (0600, never overwrites). |
| `  --retry KEY`, `--retry-failed` | Retry named Tasks (`LOST` ones included), or every Task whose latest work failed or stopped. |
| `  --only KEY`, `--tasks-per-allocation N`, `--show-scripts` | Restrict the plan; lower packing; print batch scripts (sensitive). |
| `  --budget DURATION`, `--max-links K` | Continue each Task through a chain of links until its lifetime budget is spent ([budgeted continuation](#budgeted-continuation)). |
| `validate STATE PLAN` | `sbatch --test-only` once per distinct allocation shape; exit 1 if any is rejected. |
| `submit STATE PLAN` | Submit the reviewed plan. Exit 3 if interrupted. |
| `status STATE [--offline]` | Task and allocation status; `--offline` skips Slurm. |
| `logs STATE [--task KEY] [--allocation ID] [--bytes N] [--output FILE]` | Bounded log tail; writes 0600; refuses a terminal. |
| `reconcile STATE` | Resolve every unresolved allocation from Slurm evidence; exit 75 while some still wait. |
| `cancel STATE [--task KEY]... [--allocation ID]...` | `scancel` matching allocations that Slurm lists as queued or running, newest first; a chain link brings its later links. |
| `doctor [--profile NAME]` | Check what the profile needs from its cluster, read-only; see below. Exit 1 if a check fails, 75 if one went unanswered. |

`doctor` submits nothing and writes nothing. It refuses a profile that does not fit its ceilings,
then runs every check. Each one is `pass`, `fail` (the command ran and said no, or its output is
unusable), or `unavailable` (no answer came back). A cluster that does not answer `sbatch --version`
stops there, since every other command would wait for its deadline too:

| Check | Passes when | Why Servatus needs it |
| --- | --- | --- |
| `slurm version` | `sbatch --version` reports 23.02 or later. | Every submission uses `--export=NIL`. |
| `squeue` | `squeue` exits 0 under the scrubbed environment. | Liveness comes from `squeue`, with only `PATH`, `LANG`, `LC_ALL`, and `TZ` set. |
| `MinJobAge` | `scontrol show config` reports at least 120 s. | Outcomes are read after the purge, which must outlast a cached directory listing. |
| `ProctrackType` | `scontrol show config` reports `proctrack/cgroup`. | A job Slurm no longer lists must have no steps left. |
| `work_root`, `log_root` | Each is a writable directory on the login host. | Steps start in one and write logs and run markers to the other. |
| `find -printf` | `find -printf` (GNU find) lists `log_root` with mtimes. | Outcomes and run time come from run-marker names and mtimes. |
| `sbatch --test-only` | Slurm accepts the profile's one-Task request. | It proves the partition, account, QOS, GRES, and limits go together. It says nothing about whether a job would start soon. |

`logs --task KEY` reads that Task's step log in its current allocation, `--allocation ID`
the allocation log, and both that Task's step log in that allocation; `--bytes` is 1 to 1048576
(default 65536). Key and allocation options repeat. `servatus --version` prints the version.

## Guarantees

- Intent is durable before each `sbatch`. An uncertain submission blocks its Tasks until it is
  resolved, and is recorded as never submitted only after Slurm and the run markers show nothing
  for 15 minutes; nothing is retried without an explicit decision.
- Submission executes exactly the reviewed plan, or refuses it as stale or tampered.
- A Task never runs in two allocations at once. A chain's later link runs a Task only on a handoff
  its predecessor writes after the Task's step returned.
- Campaign state is owner-only, validated on every read, atomically replaced, and synced.
- Scheduler evidence is bounded, identity-checked, and read from each Attempt's original target.
  No Slurm accounting is needed, and a missing marker counts only after Slurm forgets the job.
- A published destination is absent or complete and is never overwritten. Content is synced before
  commit and the parent after it.
- Private work survives failure and is removed only after a durable publication.
- Application meaning stays with you: Servatus never sees schemas or decides completion.

The [architecture decisions](https://github.com/edoski/servatus/blob/main/docs/adr/README.md)
explain each guarantee and the current scope, and
[SECURITY.md](https://github.com/edoski/servatus/blob/main/SECURITY.md) the trust model. To test
your launchers, pass `servatus.testing.FakeScheduler()`, an in-memory Slurm, as `connect=` to
`Campaign` or `servatus.cli.main`; its controls name commands by basename (`"sbatch"`), except the
`sbatch --version` check, which is `"ping"`.

## Non-goals

- **No job arrays or multi-node jobs.** One Task may use several GPUs on its node (`torchrun`).
- **No automatic retry, background polling, or scheduler plugins.** One native Slurm lane; a
  budgeted chain is planned and reviewed up front.
- **Not an experiment tracker, workflow engine, or secrets manager.** Task meaning belongs to you.
- **No raw Slurm options, fractional GPU requests, or cross-filesystem copies.**

See the [changelog](https://github.com/edoski/servatus/blob/main/CHANGELOG.md) for release notes.
