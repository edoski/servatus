# ADR 0005: Keep one Campaign execution authority

Status: accepted (rewritten for 0.12)

Campaign owns generic execution lifecycle facts. Applications own Task meaning, result schemas,
aggregate validation, and publication. One typed persisted state and explicit atomic transactions
own the roster and Attempt history.

## Functional core, imperative shell

Pure modules decide; one facade acts. Configuration, the codec, state and its transitions,
evidence parsing, status projection, the eligibility policy, script rendering, and planning never
spawn processes, read the clock, or touch the filesystem (except `Profile.load` reading TOML and
`Plan.save` publishing a plan file).
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
atomically (write a new file, sync it, rename it over the old one, sync the directory) through the
same pinned-descriptor primitives as publication. `update(change)` writes only when the change
returns a different state. The store caches the decoded state by file content and decodes again only
when the bytes differ.

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
Profile, selected, held, and deferred keys, retry choices (selected or deferred Tasks that resubmit
accepted work), duplicate-risk acknowledgements, packing, whether a probe was used, and a random
`nonce` from which allocation identities derive. Packing forms balanced groups in roster order
within every ceiling and `tasks_per_allocation`; `capacity(target, resources, cap=None)` is public
and is checked before any observation. `max_allocations_per_submit` bounds the batch, and eligible
work beyond it is deferred. `Plan.warnings` reports acknowledged duplicate risk and deferral.

The policy (`Hold`, `Retry`) is pure and evaluates each Task in order: `NOT_REQUESTED` (outside
`only`); `VALID` (the probe reports a valid result; valid work is never resubmitted); `UNRESOLVED`
(an Attempt naming it has intent without outcome); selected as fresh work if it was never
accepted; `SUBMITTED` if it has accepted work and resubmission was not requested for it, without
consulting the scheduler; `ACTIVE` if any accepted Attempt is queued, running, or retained;
`UNOBSERVABLE` if any accepted Attempt lacks evidence; otherwise the retry selector decides.
`Retry.FAILED` selects Tasks whose current (latest accepted) Attempt failed or was cancelled, judged
per step when step evidence exists and per allocation otherwise; other terminal Tasks stay
`SUBMITTED`. `Retry.INCOMPLETE` selects terminal Tasks whose result the probe reports missing; it
requires a probe, and planning without one raises `ConfigurationError`. Under either selector,
`UNKNOWN` evidence holds a Task as `UNOBSERVABLE`. Explicit retry keys raise `PlanRefused`, listing
every offending key, for an unknown key, a valid result, no accepted history, `UNKNOWN` evidence
without acknowledgement, or an acknowledgement of a key that was not explicitly retried (so
duplicate risk can never be acknowledged through a bulk selector). Explicit retries of active,
unresolved, or unobservable work are held, not raised. Bulk selectors never raise for individual
Tasks.

Observation is scoped: a plan observes only the accepted Attempts whose Tasks could be retried, so a
plan of fresh Tasks makes no scheduler calls. The probe is called once per operation with the
relevant Tasks.

`Plan.to_json()` serializes the Decision with its digest; `Plan.save(path)` writes it owner-only
without overwriting. `Campaign.load_plan(data)` rebuilds the Plan from the Decision and current
state and requires the same digest.

## Submission

`submit(plan)` and `validate(plan)` first rebuild the plan from its stored Decision and the current
state and compare it with the given plan: a changed campaign raises `StalePlan` and a tampered plan
`ConfigurationError`, before any claim. `validate` runs `sbatch --test-only` once per distinct
allocation shape and records nothing. Submission then requires a probe if the plan was made with
one (`ConfigurationError`), pings the scheduler (`sbatch --version`, so a broken connection records
no intent), and rechecks eligibility once for the whole submission: it probes the planned Tasks
once, observes only the Attempts of planned retries, and raises `StalePlan` unless every planned
Task is still selected. All of these failures are ordinary errors raised before any intent. For
each allocation it then checks the revision, records intent (synced), contacts Slurm outside the
lock, and records the outcome. A receipt is recorded even if an unrelated append or seal advanced
the revision; that change stops further allocations from the stale plan.

A complete batch returns `SubmitResult`. Any stop once the first intent is attempted raises
`SubmissionInterrupted` whose `result` holds confirmed receipts, unresolved allocations (with any
job observed but not durably recorded), and unattempted allocations. `KeyboardInterrupt` and
`SystemExit` propagate, leaving durable intent for recovery. Recovery uses `reconcile` (exactly one
job carrying the allocation identity, otherwise `ReconciliationError`; an already accepted
allocation returns its receipt without contacting Slurm, and one recorded as not submitted raises
`Conflict`), `mark_accepted(allocation_id, job_id, *, cluster=None)`, or `mark_not_submitted`. No
automatic retry hides ambiguity; history is never erased.

## Status

`Campaign.status(*, scheduler=True)` projects all Tasks and Attempts at one revision. Each
`TaskStatus` carries the result state, current allocation, execution state (step evidence when
known), exit code, and whether it is unresolved. Result readiness requires a sealed roster and valid
results for every Task; quiescence independently requires scheduler evidence, no unresolved
acceptance, and terminal, non-retained evidence for every accepted Attempt. `Status.counts()` has a
fixed set of keys. `scheduler=False` makes no scheduler
calls. `Status.to_json()` (`servatus.status/1`) serializes the snapshot without rereading state.

`cancel(*, tasks=None, allocations=None)` observes the chosen allocations, then `scancel`s each that
is not known to be terminal on its original route: every accepted allocation of each named Task
(`Conflict` if a Task has none), plus each named allocation (`Conflict` unless accepted). It returns
the receipts it asked to stop. `read_log(*, task=None, allocation=None, max_bytes=65_536)` reads the
allocation log, a Task's step log in its latest accepted allocation, or with both that Task's step
log in that allocation; `max_bytes` is 1 B to 1 MiB. It derives the log path from state and reads a
bounded suffix outside the store lock; log content never enters state or decisions.
