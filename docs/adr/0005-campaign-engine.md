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

Later accepted Campaign-engine work may add bounded transient scheduler and caller-result evidence
to this same authority. It must not turn Slurm completion into application validity, persist a
callback, add a durable finalized phase, or move application publication into Campaign.

This decision narrows ADR 0001 only by allowing a future ephemeral boolean result probe; application
schemas remain opaque. It supersedes ADR 0003's roster and durable submission-record shape while
retaining ADR 0003's single native lane and safety decisions. ADR 0002 publication and ADR 0004
Workspace-child behavior are unchanged.
