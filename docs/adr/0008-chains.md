# ADR 0008: Continue Tasks through chains queued at submission

Status: accepted (after 0.13)

Amends [ADR 0005](0005-campaign-engine.md) (state, planning, submission, cancellation, and the
"never two live allocations" wording) and [ADR 0007](0007-run-markers.md) (marker listing, two new
marker kinds, and a second batch-script shape).

## Context

A long training Task resumes from its own checkpoint, but its run is cut by the Slurm time limit.
Until now a person ran `plan --retry-failed` and `submit` after every limit. The controller is a
CLI on a laptop that may sleep for the whole run, so continuing cannot depend on a controller
pass. Two requirements came with it: the application counts its own budget, and continuing a Task
that already finished, with a larger budget, must be a planned operation.

Every allocation must still be planned up front, submitted with `--no-requeue` from the controller
with a durable intent first, and observed from `squeue` and run markers (ADR 0007). A Task must
never run twice at once, and done work must never rerun.

## Decision

**A plan with a budget queues a chain of allocations, called links, at submission.**
`plan --budget D` (`Continuation(budget=D)`) gives each selected Task a lifetime run-time budget.
Each packed group of Tasks becomes one chain: full `time_limit` links, then one shorter link
holding the rest of the budget in whole minutes, rounded down so the budget is never exceeded. A
rest within a minute of `signal_before_end` is dropped, since that link would be signalled as it
starts. A chain has as many links as its neediest Task, at most `max_links` (default and ceiling
`MAX_LINKS`, 32), and link j lasts as long as the neediest Task still running there needs; a Task
in it continues only for its own links, and the plan warns when packed Tasks need different
numbers of links. `submit` sends every link in
order, every chain's link 1 first, each with its own durable intent. Link j depends on links
1..j-1 with `--dependency=afternotok:...` and `--kill-on-invalid-dep=yes`: it may start only once
every earlier link ended without exit 0, and Slurm deletes it, before it allocates anything, once
any earlier link exits 0. Every link is an ordinary Attempt, so `reconcile`, the 15-minute rule,
and stray detection apply unchanged. `max_allocations_per_submit` counts every link; a plan that
would exceed it defers whole chains, never truncates one.

**A handoff marker is the one continuation decision.** A link that hands some slot on writes
`<next>.<slot>.go`, an exact name rendered from the next link's planned identity, once that slot's
step returned and the slot `continues`:

| The step | Pre-limit signal seen | End marker | Hands off |
| --- | --- | --- | --- |
| exits 0 | either | yes | no: done |
| exits 75 | either | yes | yes |
| exits c, not 0 or 75 | no | yes | no: failed |
| exits c, not 0 | yes | yes | yes (an unhandled `SIGUSR1` is 128 + 10) |
| is interrupted (TERM, HUP, INT) | yes | no | yes: time limit |
| is interrupted | no | no | no: cancelled or preempted |
| is killed (KillWait, node lost) | either | no | no |

A later link launches a slot only if its own handoff marker exists and holds a plain decimal; it
looks each name up exactly once, on a node that never looked it up before, after the previous
link ended, so the lookup cannot hit a cached negative entry and needs no wait or poll. A link
exits 0 only when every slot proved that it wrote no handoff, which lets Slurm delete the rest of
the chain. Slurm dependencies only order and prune links; the guarantee rests on the handoff.

**Only a link that hands off changes shape.** Such a link asks for `--signal=B:USR1@N`, so Slurm
signals its batch shell, whose trap writes `<allocation>.<job>.limit` and relays `SIGUSR1` to its
process group; `srun` forwards it to each step as before. Its slot subshells trap the interrupt
themselves and decide after their step returned, and it exits 0 only when no slot handed off.
Every other allocation, each plain one and every last link, keeps `--signal=USR1@N`, the plain
script, and its epilogue (exit 1 when a step failed). One flag on the rendered link, whether any
slot hands off, selects all of that. A plan without a continuation renders byte for byte as in
0.13.

**The step learns its budget and its measured run time.** Every chain step gets `SERVATUS_LINK`
and `SERVATUS_LINKS` (its 1-based position and its own planned link count), `SERVATUS_BUDGET`
(the lifetime budget in seconds, unset without one), and `SERVATUS_RUN_TIME`: the seconds the Task
ran before this allocation, across every allocation it ever had. Link 1 renders the run time the
plan measured. The batch shell records `date +%s` after its start marker, and a slot that hands
off writes `previous + now - started` into the handoff marker, so each later link reads its exact
run time from its own handoff. A step exits 0 when done and 75 (`YIELD_EXIT`) to continue.

**Run time has one evidence source: marker mtimes.** Markers are listed with
`find -printf '%T@ %p\n'`. A slot's run time in one allocation is its end (or the stop) minus the
start; a running link counts until now, clamped to its time limit; a `LOST` one counts its whole
limit; a slot never handed or never started counts nothing. The file server stamps every mtime,
so durations never mix clocks; only a running link's provisional value uses the controller's
clock, and it is marked incomplete. Run time is a projection, never stored. A plan with a
continuation records each selected Task's measured run time in its reviewed Decision (`spent`), so
`build_plan` stays a function of the Decision and state, and `submit` re-measures it and refuses
the plan (`StalePlan`) on any difference.

**The budget is a lifetime total.** Planning the same budget twice converges: a Task whose run
time leaves no room for one more link is held `SPENT`. `--retry KEY --budget D` is the only way to
extend a Task: under a budget an explicit key may name a `SUCCEEDED` Task, or one the probe
reports `VALID`, and the plan lists it as an extension (`Decision.extensions`, `extends SUCCEEDED`
or `extends VALID`). `--retry-failed --budget D` selects `FAILED` and `STOPPED` Tasks with room
left. A named key whose budget is spent refuses the plan.

**One rule finds a Task's current allocation.** It is the latest accepted Attempt that was handed
the Task: link 1 always, a later link once its handoff marker appeared, unless the marker's mtime
is later than that link's start (a predecessor cut off from Slurm finished after its successor
started, which then skipped the slot). Status, `Retry` selection, and `logs --task` all follow
it. Status labels a `STOPPED` Task `time limit`, `yielded`, `budget spent` (either of those on
the last link of a budgeted chain), or `never started`; there is no new execution state.

**Cancellation follows the chain.** `cancel --task` stops every queued or running link of the
Task, newest first, so no later link becomes eligible while an earlier one stops.
`cancel --allocation` on a link also stops every later link of its chain.

**State is schema 9, plans are `servatus.plan/3`, status is `servatus.status/3`.** Every Attempt
records its `Link` (chain, index, count, budget); a plain allocation is link 1 of 1. The invariant
requires link j > 1 to follow an accepted link j-1 of the same chain, with the same count, budget,
and Tasks, accepted before link j's intent. There is no migration.

## The invariant: never two executions of a Task

Several allocations of one Task can be listed at once, the pending links of its chain, but at most
one can run it. Link j+1 runs slot s only if `<id_{j+1}>.<s>.go` exists; only link j writes that
file, once, after slot s's step returned. Each identity is submitted once with `--no-requeue`, and
a retry or extension is held `ACTIVE` while Slurm lists any link of the Task. ADR 0005's "never
two live allocations of one Task" now reads "never two executions of a Task".

## Failure analysis

| Failure | Outcome | Safe because |
| --- | --- | --- |
| An intent or `sbatch` of link j fails | Links j+1..K are unattempted; link j is unresolved until `reconcile` | Links 1..j-1 form a shorter chain; a handoff to a link that does not exist is never read |
| `sbatch` refuses link j because link 1 was already purged | Link j is unresolved, then recorded as not submitted | Nothing runs it |
| Link j's start marker write fails | No step runs; the next link finds no handoff and exits 0 | Gating; at most one short empty allocation |
| A handoff write fails (quota, ESTALE) | The next link skips the slot; the Task reads `STOPPED` | Fails toward stopping, never toward a second run |
| Node failure or SIGKILL of link j | `LOST`; the next link finds no handoff and exits 0 | A step still running on a cut-off node is never launched again |
| Raw `scancel` of link j before the pre-limit signal | No handoff; the chain stops after one empty link | Use `servatus cancel` |
| Raw `scancel` of link j after the pre-limit signal | Reads as a time limit; the chain continues | Documented: use `servatus cancel` |
| The controller sleeps for days | Nothing | No durable step happens off the controller |

## Alternatives rejected

- **The controller submits each continuation when it sees a time limit.** Nothing advances while
  the laptop sleeps.
- **The batch script submits the next link from the compute node.** Compute-node submission can be
  forbidden, and the intent would move off the durable state into markers, which needs a second
  resolution rule for links the controller never saw.
- **`afterany` with each link globbing its predecessor's markers.** A glob is a directory read,
  which NFS caches, and it cannot tell a cancel from a time limit.
- **`--dependency=singleton` with a shared job name.** It breaks "the job name is the identity".
- **A budget per chain ("12 hours more").** Every replan would add time, and it disagrees with the
  application's own configured total.
- **Run time from `squeue %M` for running links.** A second evidence source for a value that is
  provisional anyway.

## Consequences

A chain queues up to `MAX_LINKS` pending jobs, which count toward `MaxSubmitJobs`. Link limits are
fixed at planning, so a link that yields early forfeits the rest of its limit. A nonzero exit after
the pre-limit signal continues, so a crash in the last `signal_before_end` continues too. In a
packed chain, later links hold resources for Tasks that already ended; the plan warns. Exit 75
reads as `STOPPED (yielded)` in every campaign. Status and planning need GNU `find -printf`, which
`servatus doctor` checks; the step needs `date` on its compute node.

`validate` runs `sbatch --test-only` for each later-link shape with `--kill-on-invalid-dep=yes`,
since a dependency needs job numbers that exist only at submission. Four facts only a live cluster
proves, and `scripts/cluster_e2e.py` exercises each in its chained scenario: `afternotok` over
several jobs with `--kill-on-invalid-dep=yes` deleting the rest of a chain, `--signal=B:USR1@N`
reaching the batch shell and `srun` forwarding the relayed `SIGUSR1` to its step, the exact-name
handoff lookup on NFS across nodes, and `date` on compute nodes.
