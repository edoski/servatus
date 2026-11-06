# ADR 0007: Observe work from the queue and run markers, not Slurm accounting

Status: accepted (0.13)

Supersedes the Evidence section of ADR 0003 and the recovery, duplicate-risk, and status rules of
ADR 0005.

## Context

Servatus 0.12 needed Slurm accounting (`sacct`) to report status, reconcile an unresolved
submission, and prove that one never reached Slurm. On 2026-11-06 a student cluster (Slurm 24.11.5,
`AccountingStorageType=slurmdbd`) refused every `sacct` and `sacctmgr` call with "Connection
refused" from both login and compute nodes, so Servatus could not observe any finished work there.
A research cluster (Slurm 23.11.4) on the same site has working accounting. On both, under the
scrubbed environment Servatus uses, `squeue --me --name=...` and `scancel --me --name=...` exit 0
when nothing matches, `/usr/bin/find` exists, `MinJobAge` is 300 seconds, process tracking is
`proctrack/cgroup`, and `sbatch --parsable` prints a bare job number.

Accounting answered two questions: is earlier work still running, and how did it end. It also
forced cross-checks that had nothing to do with either: submission windows anchoring history,
requeue incarnation ordering, job numbers reused by foreign jobs, and federation cluster routes.

## Decision

**Liveness comes from `squeue`.** `squeue --me --all --states=all --noheader --name=<identities>
--format=%i|%j|%T` lists every job carrying an allocation identity, finished ones included until
Slurm purges them. Every row is live: a job that is not listed cannot run anything, because its
steps die with it under `proctrack/cgroup`. A row must carry the job number the Attempt recorded;
another number is an `EvidenceConflict`.

**Outcomes come from the job itself.** The batch script writes empty marker files to `log_root`,
which the login node already reads for logs, named so they are found without a job number:

- `<allocation>.<job>.start`, created right after the script reads `SLURM_JOB_ID` and installs its
  trap. If it cannot be created, no step starts, so no start marker means no Task ran.
- `<allocation>.<job>.<slot>.<exit code>.end`, created by that slot's own subshell as soon as its
  step returns, so a slot that ends early is recorded whatever its siblings do.
- `<allocation>.<job>.stop`, created by the HUP/INT/TERM trap before it terminates the steps.

Empty files are created atomically, so a reader never sees half a marker. Marker writes use
`command :` so a failed write (quota, ESTALE) cannot exit the batch shell and orphan its steps.
The trap signals the script's whole process group, so a step's subshell dies with it and an
interrupted step records no end marker: interruption normally reads as `STOPPED`. Slurm signals the
steps and the batch shell separately, so a Task that exits on its own step's signal before the
batch shell is signalled can still record an end marker and read as `FAILED`; `--retry-failed`
selects both.

**A missing marker counts only after Slurm forgets the job.** An end marker is conclusive as soon as
it is visible. `MinJobAge` keeps a finished job in `squeue` for minutes, far longer than an NFS
client caches a directory listing (about a minute), so the one directory read that follows the purge
sees every marker: `find -H LOG_ROOT -maxdepth 1
-type f`, with `-H` because `log_root` may be a symbolic link, naming `<id>.*` exactly for up to
eight allocations and otherwise the three suffixes. Per slot:

| Queue row | Markers | Execution |
| --- | --- | --- |
| live, queued | any | `QUEUED` |
| live, running or settling | slot end | `SUCCEEDED` (0) or `FAILED` (exit code) |
| live, running or settling | no slot end | `RUNNING` |
| gone | slot end | `SUCCEEDED` or `FAILED` |
| gone | start and stop, no slot end | `STOPPED` (cancel, time limit) |
| gone | no start | `STOPPED`, never started |
| gone | start, no stop, no slot end | `LOST` (SIGKILL, node failure) |

`LOST` is deliberately uncertain: during a network partition Slurm can drop a job whose steps still
run on the cut-off node. Only an explicit `--retry KEY` resubmits it.

**Consequences in the engine.** Jobs are submitted with `--no-requeue`, so a script runs at most
once and restart counts disappear. `--comment` is gone; the job name is the identity. Holds are
`VALID`, `UNRESOLVED`, `SUBMITTED`, `ACTIVE`, and `NOT_REQUESTED`: `ACTIVE` means Slurm still lists
earlier work of the Task. `Retry.FAILED` selects `FAILED` and `STOPPED`, never `LOST`. Explicit keys
refuse live work and accept `LOST`, which removes duplicate-risk acknowledgement entirely.
Quiescence means Slurm lists no work of the campaign and no acceptance is unresolved. Cancel is
`scancel --me --name=<identity>`, which can never hit a reused job number. State moves to schema 8
(integer job numbers, no clusters, no duplicate risk) and saved plans to `servatus.plan/2`.

**Recovery is one idempotent command.** `reconcile` resolves every unresolved allocation: a queue
row records acceptance with its job, otherwise a run marker does, otherwise an intent at least 15
minutes old is recorded as not submitted, and a younger one waits (exit 75). The 15 minutes outlast
any `sbatch` still in flight after its reply was lost. `mark-accepted`, `mark-not-submitted`, and
`ReconciliationError` are deleted. As a backstop, every observation also asks `squeue` for the
identities of allocations recorded as not submitted; one that turns out live holds its Tasks
`ACTIVE`, makes the campaign non-quiescent, and `status` warns naming it and its job until it
leaves the queue. `cancel --allocation` can stop it.

## Alternatives rejected

- **Keep `sacct` and add markers only where accounting is missing.** Two evidence paths double the
  reasoning and the tests, and accounting adds nothing markers do not.
- **An `accounting = false` Target flag.** Every finished Task would stay unknown forever and proof
  of non-submission would be unsound.
- **Decide when the row first disappears.** A directory listing cached by NFS could still miss the
  last markers and turn finished work into `STOPPED` or `LOST`, which reruns it. Waiting for the
  purge makes the listing complete.
- **One exit marker per allocation, written after waiting for steps in slot order.** A slot that
  finished early would not be recorded until earlier slots ended, and an interruption in between
  would lose its result and rerun a finished Task.
- **Allow requeue and count restarts.** Incarnations bring back the ordering problem markers were
  meant to remove; `--no-requeue` makes one job one run.
- **Treat "started, never stopped, never ended" as stopped or failed.** That claims knowledge
  Servatus does not have; `LOST` asks the operator.
- **Run `sbatch --test-only` before every submission** to keep deterministic refusals from leaving
  unresolved intents. It costs a round trip per allocation and answers a different question at a
  different time; `validate` stays the explicit check and `reconcile` resolves refusals after the
  wait.
- **Markers with content** (an exit code inside the file). Reading content needs one more command
  per marker and can observe a torn write; an empty file named after its facts cannot.

## Consequences

Servatus works on any Slurm cluster where the user can run `squeue`, `scancel`, and `find`, the
login node sees `log_root`, and `MinJobAge` is nonzero and well above how long the login node may
cache a directory listing (NFS `acdirmax`, about a minute by default). With `MinJobAge=0` Slurm
never purges jobs, so their Tasks would stay `ACTIVE`; with a `MinJobAge` shorter than the cache, a
stale listing could misread finished work as `STOPPED` or `LOST`. `servatus doctor` refuses a
`MinJobAge` under 120 s and any process tracking but `proctrack/cgroup`. `STOPPED`, `LOST`, and
every retry wait for `MinJobAge`, so a cancelled Task reads `RUNNING` until the purge and
`--retry-failed` waits minutes longer than before.
Markers accumulate in `log_root` like logs; a listing that names many allocations reads every
marker there, so keep campaigns of tens of thousands of allocations in their own `log_root`. Markers
stay in `log_root` beside the logs and are as private as they are. Campaigns from 0.12 (schema 7)
and their saved plans cannot be opened.
