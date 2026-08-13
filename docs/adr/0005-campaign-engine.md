# ADR 0005: Deepen one Campaign execution authority

Status: accepted

Servatus evolves its existing `Campaign` rather than adding a workflow, runner, repository, or
backend abstraction. Campaign owns generic execution lifecycle facts. Applications retain Task
meaning, result schemas, aggregate validation, and publication.

Campaign schema 4 establishes the engine's durable roster and attempt foundations as a clean break.
The roster has two irreversible phases: `OPEN` permits exact ordered suffixes and execution;
`SEALED` permits execution but no Task change. `Campaign.open` authors or extends a roster,
`Campaign.load` only reopens it, `Campaign.tasks` returns its immutable Task tuple, and
`Campaign.seal()` atomically and idempotently ends authoring. Append and seal each advance revision,
invalidating older plans.

One ordered Attempt collection replaces parallel intent, receipt, and negative-resolution
collections. Servatus syncs the unresolved Attempt before SSH. The same logical record then receives
exactly one accepted or not-submitted outcome. Each Attempt retains its allocation identity, ordered
Task and retry keys, Campaign revision, target and resource lineage digests, plan and script digests,
effective allocation totals, exact nonsecret command, and reconciliation window. Reconciliation,
explicit resolution, and retry append or update only these generic execution facts; history is never
rewritten or collapsed.

The existing native OpenSSH/Slurm/Apptainer lane, balanced single-node packing, target ceilings,
submission cap, intent-before-contact rule, ambiguity handling, explicit retry, reconciliation,
validation, and status projections remain. Plan schema 3 remains current for this state slice.

`Campaign.inspect()` adds one immutable, revision-bound view to this same authority. It invokes an
optional synchronous boolean result probe exactly once per Task and queries only exact accepted
Attempt identities through bounded native `squeue` and `sacct` calls. Each Attempt receives one
server-side single-job queue request plus duplicate-preserving accounting selected and verified by
its allocation-derived job name/comment. One original accounting record inside the reconciliation
window anchors the accepted Attempt; strictly later records with that exact immutable identity are
requeue incarnations, and the unique latest incarnation owns accounting evidence. Slurm Job-ID and
cluster reuse therefore cannot collapse Attempt evidence. An active row is attributable only with
that accounting anchor. The exact native invalid-job response means no active row and does not
suppress accounting; unrelated, multiply plausible, or noncanonical accounting history fails
closed. Scheduler and reconciliation commands use a fixed C locale and UTC timezone.
Probe and scheduler operations run outside the Campaign lock. Inspection then rejects any Campaign
revision change. Their answers are transient, time-stamped, redacted, and never stored. Receipt
identity exists once on Attempt evidence rather than being copied into allocation evidence.

`Campaign.read_log()` adds one separate diagnostic operation over the same accepted Attempt
authority. It snapshots the validated immutable target, receipt, ordered Task slot, and derived
allocation-bound log path under the Campaign state lock, then releases the lock before one fixed
bounded OpenSSH suffix read. The public frozen `LogSnapshot` carries arbitrary bytes outside its
`repr`, a truncation flag, and observation time. Callers cannot choose a host, path, Job ID, or
remote command. Failures expose one redacted `ObservationError` without partial content or unsafe
exception chaining.

Every Attempt remains visible. The latest accepted Attempt owns current Task execution, unresolved
acceptance dominates that projection, and quiescence requires terminal evidence for every accepted
Attempt. Result readiness instead requires a sealed roster and valid immutable caller results for
every Task. Missing Slurm accounting therefore yields unknown execution without overriding valid
application results. Scheduler failure, timeout, overflow, malformed or partial output, unrelated
identity, or conflicting evidence aborts the complete inspection.

The observation seam does not turn Slurm completion into application validity, persist a callback,
add a durable finalized phase, authorize retry, or move application publication into Campaign.
Diagnostic log content likewise never enters Campaign state, views, plans, or records and has no
effect on readiness, quiescence, planning, retry, reconciliation, or result validity. There is no
log decoder, parser, range/offset protocol, follow mode, poller, cache, or public transport adapter.

This decision narrows ADR 0001 only by allowing an ephemeral boolean result probe; application
schemas remain opaque. It supersedes ADR 0003's roster and durable submission-record shape while
retaining ADR 0003's single native lane and safety decisions. ADR 0002 publication and ADR 0004
Workspace-child behavior are unchanged.
