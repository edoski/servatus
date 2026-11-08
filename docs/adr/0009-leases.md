# ADR 0009: Hold an allocation as a lease and start Tasks in it as steps

Status: accepted (after 0.13)

Amends [ADR 0005](0005-campaign-engine.md) (state, planning, submission, reconciliation,
cancellation), [ADR 0007](0007-run-markers.md) (a marker kind, a batch-script shape, and step
observation), and [ADR 0008](0008-chains.md) (a lease may be a chain).

## Context

On a site whose quality of service counts running jobs, not steps, every cancel and resubmit gives
the GPUs back to the queue, and other users take them within a minute. The motivating account may
run 3 jobs on 4 GPUs for at most 72 hours. The operator wants to hold GPUs for the full limit and,
from a laptop, start, stop, and watch arbitrary work inside that allocation.

Servatus already owns everything such a held allocation needs except the "later": durable intent
before scheduler contact, one Attempt history per Task, one step renderer and its launchers, run
markers read from one directory listing, liveness from `squeue`, run time from marker mtimes,
holds and retry selection, chains, cancel, and logs. An earlier unreleased design built leases as
a second subsystem with its own state on the cluster, a Python dispatcher, its own statuses, and
eight verbs. That duplicated every rule the Campaign encodes.

The constraints: one implementation of each thing, one source of truth, never two executions of a
Task, no Task starting after its lease stopped taking work, convergence after any crash, nothing a
user supplies becoming shell code, owner-only files, no process on the login node, and nothing but
POSIX `sh` on compute nodes.

## Decision

**A lease is an Attempt with no Tasks; a Task started in it is a step, an Attempt of one Task.**
An Attempt records `slots` and `within`, and `Kind` names what it holds:

| Kind | Task keys | `slots` | `within` | Job |
| --- | --- | --- | --- | --- |
| allocation | one or more | none | none | its own |
| lease | none | N held | none | its own |
| step | exactly one | none (one slot) | the lease link | the lease's |

A lease holds N copies of its Profile's per-Task Resources, exactly as an allocation of N Tasks
would ask for them, so `capacity` and every ceiling apply. A step uses exactly one slot, the
Profile's per-Task Resources, and runs as one exact `srun` step of the lease's job; a run that needs
two GPUs uses a lease whose Profile has `gpus = 2`. The invariant requires a step to name a lease
accepted before the step's intent, to use the lease's Profile, to hold no slot count of its own,
and, once accepted, to carry the lease's job. A lease chain keeps its slots on every link. Every
other rule (intent, roster order, unresolved overlap) applies unchanged, so `history`, the current
allocation, run time, retry, holds, and logs all work: a step is slot 0 of its own Attempt.

**A lease is planned and submitted like an allocation.** `Campaign.hold(profile, slots=N)` (`plan
--hold N`) plans one lease that selects no Task; `Continuation(max_links=K)` queues K lease links,
each depending on the earlier ones (`afternotok`, `--kill-on-invalid-dep=yes`), as in ADR 0008. A
lease needs `signal_before_end` and refuses a budget. It always asks for the pre-limit signal in
its batch shell (`--signal=B:USR1@N`), and its batch script never exits on its own, so the next
link starts whenever one ends, unless the chain is cancelled through Servatus.

**One slot per step, because Slurm refuses a larger one.** The lease is submitted as
`--ntasks=N --cpus-per-task=C`, so a step asking for `--cpus-per-task=K*C` exceeds the job's
per-task CPUs, and Slurm refuses it at once instead of letting it wait for room. On the university
cluster (Slurm 24.11.5, `select/cons_tres`, `CR_CPU_MEMORY`), a two-slot step in a lease of two
1-CPU slots, one of them busy, exited 1 with `Job step's --cpus-per-task value exceeds that of job
(2 > 1). Job step may never run.` and then `Unable to create step for job …: Memory required by
task is not available`. One-slot steps behave as designed: with every slot busy, the next step
waited (`step creation temporarily disabled, retrying (Requested nodes are busy)`) and started once
a slot freed.

**Contact for a step is an order.** `Campaign.plan(lease=ID)` (`plan --lease ID`) picks
the lease's current link: the first accepted link of the lease's chain that Slurm lists, has not
ended, and has no limit marker. Any link's id names the chain. Each selected Task becomes a step,
link 1 of 1 of its own chain; steps are not Slurm jobs, so they never count against
`max_allocations_per_submit` and are never deferred. `submit` records the step's intent, writes its
order, `<log_root>/<lease>.d/<step>.run`, with one fixed shell program (temporary name, then
rename, into an owner-only directory it refuses when it is a symbolic link or not this user's own),
and records the lease's job as the step's. `submit` re-observes the lease first and refuses the
plan (`StalePlan`) once the link ended or reached its pre-limit signal.

**One claim name decides every order.** The lease takes an order, `cancel` withdraws it, and
`reconcile` records it as not submitted only by creating `<lease>.d/<step>.claim`, a symbolic link
made without replacing: `taken` for the lease, `withdrawn` for the client. Exactly one creation
succeeds, and `readlink` tells the loser, and a winner whose reply was lost, who won. The lease
starts a step only after its own claim won. The client writes `<step>.<job>.void` only after its
claim won, so the void marker is evidence of the decision, never a participant in it. A start or
end marker of a step whose claim the client holds is an `EvidenceConflict`.

**The lease's batch script is POSIX `sh` and never waits for a step.** It shares the Campaign's
prelude, start marker, and interrupt trap (`<lease>.<job>.stop`, `TERM` to the process group, exit
1). Every 5 seconds it lists `*.run` in its order directory, and for each new 24-hex name, unless
the pre-limit signal has arrived (checked before every order), it claims the order and starts it in
a background subshell. That subshell sources the order, which defines the step's run time and one
`launch` function rendered by the Campaign's step renderer, runs it, waits for its `srun` whatever
signal arrives, and writes `<step>.<job>.0.<code>.end`. Slurm's step scheduler places steps: an
`srun --exact` that finds every slot busy waits inside the allocation until one frees up, so the
script does no accounting and the compute node needs no Python. Slurm, not Servatus, chooses which
waiting step starts; three steps ordered together on the university cluster did not start in roster
order. A waiting step can start up to about two minutes after a slot frees, because `srun` backs
off between retries of step creation (measured twice on the university cluster: about 2 minutes
from one step's end to the next one's start).

**The step writes its own start marker.** The order's `srun` runs `sh -c 'command : > "$1" || exit
1; shift; exec "$@"'` before the launcher, so `<step>.<job>.start` appears only once Slurm gave
the step room. A step waiting for room reads `QUEUED` and accrues no run time. A step that cannot
write its start marker exits 1 before the Task runs.

**A stop request is a marker the lease acts on.** `cancel` writes `<step>.<job>.stop`, then
contends for the claim. If the client's claim wins, the step never runs. If the lease took the
order, its per-step watcher sees the marker: a step that started gets `SIGUSR1` on its `srun`,
which `srun` forwards as is, and `SIGTERM` after `signal_before_end`, which `srun` turns into
`SIGKILL`; a step still waiting for room gets `SIGTERM` at once and never starts. No Slurm command
runs on the compute node. The watcher ignores `SIGUSR1` and the step's shell catches it, so the
relayed pre-limit signal kills neither, while `srun`, started from a shell that catches it rather
than ignores it, keeps forwarding it.

**After the pre-limit signal the lease starts nothing more.** Its trap writes `<lease>.<job>.limit`
and relays `SIGUSR1` to its process group, as a chain link that hands off does. It checks the limit
before every claim and again after one, so it starts no step once the signal arrived. Orders it has
not taken stay unclaimed and keep their Tasks held, since the lease could still take one claimed
as the signal came: `cancel` withdraws them, and they can then start in the next link.

**A step is observed through its lease.** `observe_step` combines the lease's queue row and limit
marker, the step's own start and end markers, the earlier of the step's stop request and the
lease's stop marker, and whether the order was withdrawn. Evaluated top to bottom:

| Lease row | Step markers | Execution |
| --- | --- | --- |
| any | void | `STOPPED (never started)` |
| queued | any | `QUEUED` |
| any | end 0 | `SUCCEEDED` |
| any | end 75 | `STOPPED (yielded)` |
| any | end c, limit at or before it | `STOPPED (time limit)` |
| any | end c, stop at or before it | `STOPPED` (`never started` without a start) |
| any | end c | `FAILED` |
| settling | no end | `ENDING` |
| running | no start | `QUEUED` (ordered, or waiting for room) |
| running | start | `RUNNING` |
| gone | no start | `STOPPED (never started)` |
| gone | start, stop | `STOPPED` (`time limit` with a limit marker) |
| gone | start, no stop | `LOST` |

A step stops being live, and its Task becomes eligible for retry, as soon as its end marker or its
void exists, while the lease runs on; it is also not live once Slurm forgets the lease. A stopped
step's Task can start again once its end marker appears. Run time counts from the step's own start
marker to its end; a step asked to stop counts until now until its end marker appears, and one the
lease's interrupt ended counts until the lease's stop marker. An allocation's table in ADR 0007 is
unchanged.

**`reconcile` resolves a step at once.** It contends for the claim: when the client's claim wins,
the step was not submitted and a late order is never taken; when the lease had taken the order,
the step was accepted with the lease's job. The claim, not time, makes the decision safe, so the
15-minute rule does not apply. An order that was written but not yet taken is withdrawn, so its
Task must be started again.

**Cancellation follows the kind.** `cancel --task KEY` or `--allocation STEP` stops each live step
inside its lease, which runs on. `cancel --allocation LEASE` cancels that link and every later link
of its chain, newest first, which ends every step in them.

**Steps get the Campaign's step interface.** An order renders through the same function as a
Campaign step, with a link 1 of 1 chain: `SERVATUS_TASK_KEY`, `SERVATUS_ALLOCATION_ID` (the step's
id), `SERVATUS_SLOT=0`, `SERVATUS_JOB_ID` and `SERVATUS_JOB_END_TIME` (the lease's job and its
deadline), `SERVATUS_LINK=1`, `SERVATUS_LINKS=1`, `SERVATUS_RUN_TIME` (measured at planning), and
`SERVATUS_BUDGET` when the step plan carries a budget, under which a Task whose run time leaves no
room is held `SPENT`. A step plan requests nothing new from Slurm, so its `Plan.totals` are zero.

**One command starts work.** `servatus start STATE TASKS.jsonl --lease ID` registers the file's
unseen Tasks, refusing a changed Task under an existing key as `ensure` does, plans exactly the
file's keys into the lease, and submits them. A key that already ran is held unless it is named
with `--retry KEY` or selected by `--retry-failed`. Stopping is `cancel --task KEY`, and watching is
`status` and `logs --task KEY`.

**State is schema 10, plans are `servatus.plan/4`, status is `servatus.status/4`.** Attempts gain
`slots` and `within` (a lease has `slots`, a step `within`); the Decision gains the same two fields
(a lease plan has `slots`, a step plan `within`); `PlannedAllocation` and `AttemptStatus` gain the
same two fields and `Observation` gains `step`. There is no migration.

## The invariant: never two executions of a Task

A new step of a Task is planned only when every earlier Attempt of it is not live. A step stops
being live only on evidence that it can no longer run: its end marker, written after its `srun`
returned; its void, written after the client's claim won, which the lease's claim can then never
win; or its lease gone from Slurm, whose steps die with it under `proctrack/cgroup`. Each order is
taken at most once, by the one lease link whose directory it is in.

## Failure analysis

| Failure | Outcome | Safe because |
| --- | --- | --- |
| Client dies after a step's intent, before its order | Step unresolved, Task held | `reconcile` claims the order and records it not submitted |
| Order written, reply lost | Unresolved | `reconcile`: accepted if the lease took it, else withdrawn |
| A stale order write lands after `reconcile` won the claim | The lease's claim fails; it never runs | One claim name, created once |
| The lease's or the client's claim reply is lost | `readlink` shows the winner | Ownership is proven by the link, not by the reply |
| A step cannot write its start marker | It exits 1 before the Task runs | No start marker means no Task ran |
| An end marker write fails | `RUNNING` while the lease lives, then `STOPPED` or `LOST` | Fails toward holding the Task, never toward a second run |
| The pre-limit signal arrives with orders waiting | They stay unclaimed; read `QUEUED` until the lease ends, then `STOPPED (never started)` | The limit is checked before every order |
| Raw `scancel` of a lease | The trap stops it; steps read `STOPPED`; a later link starts | Cancel through Servatus |
| Node failure or SIGKILL of the lease | Started steps read `LOST`, others `STOPPED (never started)` | ADR 0007 semantics |
| The laptop sleeps for days | Running steps go on; waiting orders start as slots free | No durable decision happens off the controller |

## Alternatives rejected

- **A second subsystem beside `Campaign`.** It needs its own state, statuses, reconcile, and verbs,
  a parallel copy of every rule the Campaign already encodes.
- **`srun --jobid` from the login node.** A synchronous `srun` lives as long as its step, so it
  needs a detached process on the login node, and a site's cleanup of that process kills the Task.
  `--overlap` gives up GPU isolation.
- **`srun --jobid --async`.** Beta in Slurm 26.05, needs the step manager, forwards no stdin, and
  reports exit codes only through accounting; the reference clusters run 23.11 and 24.11.
- **Steps of K slots.** An `srun` asking for K times the per-task CPUs exceeds the lease's
  `--cpus-per-task`, and Slurm refuses it instead of queueing it (see Decision).
- **A Python dispatcher that packs shares itself.** Needed only to reap steps and account for
  resources; per-order subshells and `srun --exact` remove both reasons.
- **Withdrawal by renaming the order plus a void created exclusively.** The create was not
  serialized with the lease's check-then-rename, so `reconcile` could record a running step as not
  submitted. One claim name contended by every side closes that race.
- **Stopping steps with `squeue` and `scancel` from the compute node.** It depends on site policy
  for Slurm commands inside jobs and on an undocumented signal sequence for a cancelled step, and
  it cannot stop an `srun` still waiting for room.
- **Carrying a running step into the next lease link.** Budgeted chains already continue one run
  across allocations; a step runs in the link it was ordered into.

## Consequences

Slots are homogeneous, and a step is one slot: a CPU-only Task in a GPU lease takes a slot's GPU
with it, and a Task that needs more than one slot needs a lease of a larger Profile. Slurm, not
Servatus, chooses which waiting step starts when a slot frees, and the start can lag the free slot
by up to about two minutes of `srun` retry backoff. Starting and stopping take up to a poll interval
plus the filesystem's attribute cache on the compute node, which on NFS is about a minute. Task keys
stay immutable, so changed work is a new key, and a lease Campaign stays appendable. A lease holds
its resources whether or not a step runs. Steps do not carry over to the next lease link; order new
steps into the next link, or use budgeted chains for single runs longer than one allocation. Orders
and claims stay in the lease's owner-only order directory after it ends, as logs and run markers
stay in `log_root`; an order holds its Task's arguments, environment, and stdin, so remove
`<log_root>/<lease>.d` once the lease is gone if they are sensitive.

`scripts/cluster_e2e.py` exercises the lease on a live cluster. A CPU run on the university cluster
showed one-slot steps waiting for a free slot and starting once one freed, stopping through
`SIGUSR1` (exit 75), claims, `test -O`, order pickup, and the held allocation outliving its steps.
Still unproven there: how long an order takes to become visible on the compute node on every
filesystem; that a GPU step gets exactly its GRES and its own `CUDA_VISIBLE_DEVICES`; what a
waiting `srun` does on `SIGTERM`; that a no-replace symbolic link created from a login node and a
compute node at the same moment lets exactly one win; that a held GPU stays held while steps come
and go; and that a lease chain's next link starts when one ends.
