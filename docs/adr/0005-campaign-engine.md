# ADR 0005: Keep one Campaign execution authority

Status: accepted (rewritten for 0.12)

Campaign owns generic execution lifecycle facts. Applications own Task meaning, result schemas,
aggregate validation, and publication. One typed persisted state and explicit atomic transactions
own the roster and Attempt history.

## Functional core, imperative shell

Pure modules decide; one facade acts. Configuration, the codec, state and its transitions,
evidence parsing, status projection, the eligibility policy, script rendering, and planning never
spawn processes, read the clock, or touch the filesystem (except `Profile.load` reading TOML).
`Campaign` is the only imperative shell: it composes the store, the scheduler, the probe, and the
clock, and runs remote work and application callbacks outside store locks. Tests drive the shell
through `servatus.testing.FakeScheduler`, passed as `connect`, instead of patching remote calls.

## State

`State` validates its complete invariant whenever it is constructed, in time linear in Tasks plus
attempted Task references, so an invalid state can be neither written nor read. Transitions are
pure functions returning a new `State`: `create`, `append`, `seal`, `record_intent`, and
`record_outcome`. `record_outcome` is idempotent for an identical repeat, raises `Conflict` for a
conflicting outcome, and `NotFound` for an unknown allocation.

The document is schema 7. Each Attempt references its Profile by a short digest of the Profile's
canonical encoding, and each distinct Profile is stored once, so long histories do not repeat
identical targets. Decoding requires the exact canonical encoding; any defect raises
`CorruptState`. Schema 6 and earlier are rejected; there is no migration.

The store is an owner-only directory with a `.lock` file (`flock`) and `campaign.json`, replaced
atomically (write a new file, sync it, rename it over the old one, sync the directory) through the same pinned-descriptor
primitives as publication. `update(change)` writes only when the change returns a different state.
The store caches the decoded state by file content and decodes again only when the bytes differ.

## Authoring

`Campaign.create(path, tasks, *, appendable=False)` creates a sealed roster by default and raises
`Conflict` if state exists. `Campaign.open(path)` raises `NotFound` for missing state.
`Campaign.ensure(path, tasks, *, appendable=True)` creates (making parent directories) or opens; a
registered Task must be identical by key (`Conflict` names the first mismatch) and unseen Tasks are
appended in order. `append` accepts only new keys; `seal` irreversibly ends authoring. All accept
an optional `probe`, `connect`, and `clock` bound to the handle; callbacks are never serialized.

## Planning

`Campaign.plan(profile, *, retry=(), allow_duplicate_risk=(), only=None,
tasks_per_allocation=None)` produces a `Plan` from a `Decision`: Campaign identity and revision,
Profile, selected keys, retry choices, duplicate-risk acknowledgements, packing, and a random
`nonce` from which allocation identities derive. Packing forms balanced groups in roster order
within every ceiling and `tasks_per_allocation`; capacity is checked before any observation.
`max_allocations_per_submit` bounds the batch, and eligible work beyond it is deferred.

The policy (`Hold`, `Retry`) is pure. A Task is held as `VALID` (the probe reports a valid result),
`ACTIVE` (Slurm retains earlier work), `UNRESOLVED` (intent without outcome), `FINISHED` (terminal
earlier work not selected for retry), `UNOBSERVABLE` (unknown evidence without acknowledgement), or
`NOT_REQUESTED` (outside `only`). `Retry.FAILED` selects Tasks whose current Attempt failed or was
cancelled, judged per step when step evidence exists and per allocation otherwise.
`Retry.INCOMPLETE` selects every terminal accepted Task without a valid result and requires a
probe. Explicit retry keys raise `PlanRefused`, listing every offending key, for an unknown key, a
valid result, no accepted history, unknown evidence without acknowledgement, or an acknowledgement
without retry. Explicit retries of active, unresolved, or unobservable work are held, not raised.
Bulk selectors never raise for individual Tasks.

Observation is scoped: a plan observes only the accepted Attempts whose Tasks could be retried, so a
plan of fresh Tasks makes no scheduler calls. The probe is called once per operation with the
relevant Tasks.

`Plan.to_json()` serializes the Decision with its digest; `Plan.save(path)` writes it owner-only
without overwriting. `Campaign.load_plan(data)` rebuilds the Plan from the Decision and current
state and requires the same digest.

## Submission

`submit(plan)` and `validate(plan)` first rebuild the plan from its stored Decision and the current
state and compare digests, so a stale or tampered in-memory plan is refused (`StalePlan`) before any
claim. Submission then pings the scheduler (`sbatch --version`), so a broken connection records no
intent; observes the relevant Attempts once for the whole submission; and probes once. For each
allocation it checks the revision, records intent (synced), contacts Slurm outside the lock, and
records the outcome. A receipt is recorded even if an unrelated append or seal advanced the
revision; that change stops further allocations from the stale plan.

A complete batch returns `SubmitResult`. Any stop raises `SubmissionInterrupted` whose `result`
holds confirmed receipts, unresolved allocations (with any job observed but not durably recorded),
and unattempted allocations. `KeyboardInterrupt` and `SystemExit` propagate, leaving durable intent
for recovery. Recovery uses `reconcile` (exactly one matching job, otherwise
`ReconciliationError`), `mark_accepted(allocation_id, job_id, *, cluster=None)`, or
`mark_not_submitted`. No automatic retry hides ambiguity; history is never erased.

## Status

`Campaign.status(*, scheduler=True)` projects all Tasks and Attempts at one revision. Each
`TaskStatus` carries the result state, current allocation, execution state (step evidence when
known), exit code, and whether it is unresolved. Result readiness requires a sealed roster and valid
results for every Task; quiescence independently requires scheduler evidence, no unresolved
acceptance, and terminal evidence for every accepted Attempt. `scheduler=False` makes no scheduler
calls. `Status.to_json()` (`servatus.status/1`) serializes the snapshot without rereading state.

`cancel(*, tasks=None, allocations=None)` cancels current accepted allocations on their original
routes. `read_log(*, task=None, allocation=None, max_bytes=65_536)` derives the log path under the
store lock and reads a bounded suffix outside it; log content never enters state or decisions.
