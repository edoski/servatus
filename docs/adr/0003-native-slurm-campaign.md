# ADR 0003: Own one native Slurm campaign

Status: accepted (revised for 0.12)

Servatus supports one concrete lane: an unprivileged user runs absolute Slurm executables, either
through OpenSSH from a workstation or directly on a login node, and each Task starts through one of
two launchers. Campaign owns balanced single-node allocations, complete batch scripts, and durable
submission records. Application meaning and result validity remain with the caller. ADR 0005
defines Campaign state and the public workflow.

The package invokes stable command-line seams directly. There is no scheduler adapter or plugin
interface without a second proven production lane, and no remote Python runtime is required to
submit or observe work.

## Transport

`Transport` is the only process-spawning seam: `run(argv, *, stdin=b"", max_stdout=...)` returns
the exit status and both output streams. `connect(target)` returns `Ssh(target.host)` when the
Target has a host and `Local()` otherwise; callers and tests may supply their own `connect`.

- `Ssh` runs `ssh -T -o BatchMode=yes -o LogLevel=ERROR HOST REMOTE`. The remote command prints a
  per-call random marker to stdout and stderr, then `exec`s `/usr/bin/env -i PATH=... LANG=C
  LC_ALL=C TZ=UTC` with the argv. Output before the markers (login banners, shell-startup noise) is
  discarded; a missing marker means the command did not run as intended and is `Unavailable`.
  Batch mode means Servatus never prompts; connection reuse and multi-factor login belong in the
  user's ssh_config (`ControlMaster`/`ControlPersist`). The local `ssh` client gets a fixed `PATH`
  and C locale plus only `HOME`, `LOGNAME`, `USER`, `SSH_AUTH_SOCK`, `KRB5CCNAME`, and
  `KRB5_CONFIG` from the caller, enough for its configuration, agent, and Kerberos credentials.
- `Local` runs the argv with the same scrubbed environment and no shell.

Every operation has a 30-second deadline, at most 32 arguments, 16 KiB of command text, 4 KiB per
field, and 1 MiB per output stream (plus 64 KiB for SSH noise before the marker); streams are
drained concurrently and a failing child is killed and reaped. Once the child exits, the runner
reads what is already buffered and stops waiting for end-of-file, because a descendant (an OpenSSH
`ControlPersist` master started by this call) may hold the pipes open indefinitely. Spawn, timeout,
overflow, missing-marker, and connection failures, and scheduler commands that fail where success is
required, all surface as `Unavailable`: one error boundary. `check_command(argv)` applies the bounds
purely, so planning rejects an oversized `sbatch` request before any intent exists.
`servatus.campaign.ping(target)` runs `sbatch --version` through this seam for `servatus doctor`,
and every submission runs the same check first; a failure reports Slurm's first stderr line.

## Batch script and launchers

Allocation resources equal the sum of concurrent exact steps. Servatus never infers node capacity,
escalates a request, emits job-level exclusivity, or accepts raw Slurm options. Wall time is
rounded up once to whole minutes. Target ceilings prevent user mistakes but do not replace cluster
policy.

Each allocation starts one `srun --exclusive --exact --nodes=1 --ntasks=1` step per Task, named
`servatus-<allocation_id>-<slot>` with `--job-name`. Each step's stdin is a single-quoted `printf`
literal piped into `srun`: printable ASCII verbatim, every other byte as `\ooo`. The script needs
no scratch directory, `mktemp`, `base64`, or cleanup. Every step receives the Task environment plus
`SERVATUS_TASK_KEY`, `SERVATUS_ALLOCATION_ID`, `SERVATUS_SLOT`, `SERVATUS_JOB_ID` (from
`$SLURM_JOB_ID`), and `SERVATUS_RESTART_COUNT` (`${SLURM_RESTART_COUNT:-0}`).

- **Apptainer launcher:** `apptainer exec --cleanenv` with `--bind WORK_ROOT:WORK_ROOT`, one
  `--bind` per Target bind, `--pwd WORK_ROOT`, and `--nv` for GPU work, then the image and the
  Task's arguments. `exec`, not `run`: the Task's `args` are the program and its arguments
  whatever the image's runscript does, and a relative `args[0]` resolves on the container's `PATH`.
  An Apptainer Task with empty `args` is a `ConfigurationError` at planning. Environment values are
  passed as `APPTAINERENV_NAME=<shell-quoted value>` assignments, never `--env`, so values
  containing commas or equals signs survive intact.
- **Direct launcher:** `env -i ... NAME=VALUE` runs the Task's `args[0]`, which must be an absolute
  path without `=` (`ConfigurationError` at planning otherwise), from `work_root`.

GPU steps forward Slurm's step-local `CUDA_VISIBLE_DEVICES` (failing the step when it is unset) and
set `CUDA_DEVICE_ORDER=PCI_BUS_ID`; site configuration owns device isolation. When the Resources set
`signal_before_end`, each step receives `--signal=USR1@<seconds>`. The batch records each step's
pid; its interrupt trap signals only those recorded pids. It waits for every started step and
aggregates failure into the allocation's exit status.

Allocation output goes to `log_root/<allocation_id>-%j.out` and each Task's combined output to
`log_root/<allocation_id>-%j-<slot>.out`. The immutable allocation identity prevents reused job
numbers from aliasing Attempts. Paths that these commands would reinterpret are rejected when the
Target is built: `%` in `log_root` (Slurm filename patterns), `,` or `:` in `work_root` (bind
syntax), and `:` in the Apptainer image (Apptainer reads it as a URI such as `docker://`).

## Evidence

Observation batches at most 16 jobs per query, grouped by each Attempt's original Target and
cluster; reused job numbers are queried separately. Each batch runs `squeue`, anchored
`sacct --allocations --duplicates` history, and a step query. Replies are bounded to 4096 lines and
4096 bytes per field, and must be complete, strict UTF-8 rows without control characters. `squeue`'s
exact "Invalid job id specified" reply means none of the queried jobs is active. One state table
normalizes Slurm states to `QUEUED` (including `EXPEDITING`), `RUNNING`, `SUCCEEDED`, `FAILED`,
`CANCELLED`, or `UNKNOWN`; unlisted states such as `REVOKED` are `UNKNOWN` and retained.

Requeue history is anchored: only the first `sacct` row must fall in the submission window (intent
time plus or minus one hour); each later incarnation must have a strictly increasing submit time on
the same cluster. Within one incarnation, when `squeue` and `sacct` disagree, the more advanced
state wins (terminal over active, running over queued); a terminal queue row without accounting is
`UNKNOWN`. Contradictions confined to one allocation's job number become a per-allocation
`problem`: that allocation is `UNKNOWN` and retained, which withholds only its Tasks instead of
aborting the whole observation. That includes a queried job number now held by a foreign job (a
row whose name or comment is not the allocation identity, after job-number reuse). Reply-level
defects (partial or oversized output, malformed rows, rows for job numbers that were not queried,
rows from another cluster) still raise `EvidenceConflict` for the query. Held or requeued work,
including `SPECIAL_EXIT`, is retained and blocks retry.

**Step evidence.** One additional `sacct` query (without `--allocations`) maps step rows named
`servatus-<id>-<slot>` to per-Task `StepEvidence`: one entry per slot, `None` when that step was
not found. A failure of that query leaves `steps` empty (step evidence unavailable, not absent); it
never aborts observation. How a Task's execution follows from step and allocation evidence is
ADR 0005's status rule.

**Identity search.** `Scheduler.find` lists every job carrying an allocation's identity, queued or
running (`squeue --name`) or accounted within the submission window (`sacct --name`). An empty
answer proves absence; a failed command is `Unavailable`, and a malformed or unrelated row, or one
job number on several clusters, is `ReconciliationError` because it proves nothing. `identify`
(used by `reconcile` and `mark_accepted`) requires exactly one such job.

**Cancel.** `scancel --name=servatus-<id> JOB` targets the allocation's job on its original route
and cluster, so a reused job number is never cancelled; a job that already ended or aged out counts
as cancelled. Cancellation is a request; retry stays blocked until evidence shows the work
terminal. `Campaign.cancel` tries every chosen allocation and reports failures together.

**Logs.** Log reads derive one path from a validated accepted Attempt, slot, and job number and
`tail` a bounded suffix. No caller-supplied remote path or command is accepted; the path stays
contained in the Attempt's `log_root`.

## Recorded production acceptance

This section predates 0.12. It records the live gate of the 0.11 candidate below; the 0.12
transport, launcher, payload, and step-evidence changes have not yet passed an equivalent gate.

The following evidence concerns the tested candidate and site, not a live acceptance of every later
release. On 2026-08-10, candidate `0c454bd38da4f3d5b0ba4f0777b708f8a2eb011c` passed the live gate as
an unprivileged user-side client on Slurm 23.11.4. The site used `select/cons_tres` with
`CR_CPU_MEMORY`, task cgroup and affinity plugins, `/usr/bin/ssh`, Slurm commands under `/usr/bin`,
and `/usr/bin/apptainer`. Bounded validation and jobs 44592–44595 proved CPU-only, one-GPU,
one-process/two-GPU, byte-exact argv/stdin, exact requested and allocated TRES, receipt durability,
and sibling failure aggregation. Job 44598 proved four packed one-GPU steps with four distinct GPU
UUIDs beginning within four milliseconds.

The first four-pack also established the topology boundary: on this SMT2 site, one requested Slurm
CPU represented one logical thread while each exclusive step occupied a physical core, so only two
one-CPU steps placed simultaneously. The accepted four-pack used `cpus_per_task=2` and requested
exactly eight CPUs. Callers must describe that topology truthfully. Servatus continues to preserve
resource arithmetic and binding; it does not auto-inflate CPUs, disable affinity, or add raw Slurm
options.
