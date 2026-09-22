# ADR 0005: Keep one Campaign execution authority

Status: accepted

Campaign owns generic execution lifecycle facts. Applications own Task meaning, result schemas,
aggregate validation, and publication. One typed persisted state and explicit atomic transactions
own the roster and Attempt history. Campaign and plan documents use schema 5 with no migration or
compatibility decoder. Remote work and application callbacks run outside store locks.

## Authoring and configuration

`Campaign.create(path, tasks, appendable=False)` creates a sealed fixed roster by default.
`Campaign.load(path)` only opens existing state. Both accept an optional `probe` bound to the
handle for inspection, planning, and submission; callbacks are never serialized. For an appendable roster, `campaign.append(tasks)`
accepts only a new ordered suffix and `campaign.seal()` irreversibly ends authoring. Registered
Tasks never change. Append and seal advance revision; execution is valid before sealing.

A Profile supplies a nonbinding label, Target, and homogeneous Resource request for one plan.
Every Attempt retains those resolved values. Later plans may change routes or resources without
rewriting historical Attempts; observation, reconciliation, and log reads use each original route.
`SERVATUS.toml` has named profiles and an optional default. TOML syntax and unknown keys are checked
throughout the document, while semantic validation concerns only the selected profile. There is no
inheritance, environment fallback, parent-directory search, or global configuration store. A sole
profile selects itself. Target and resource constructors are keyword-only, absent site settings
default to `None`, GPU requests default to zero, and task stdin defaults to empty bytes. Argument
and partition sequences freeze to tuples. Task/resource/time ceilings remain explicit for packing;
submission and script bounds default to one allocation and 1 MiB without additional policy objects.

## Planning and submission

`Campaign.plan(profile, retry=(), allow_duplicate_risk=(), tasks_per_allocation=None)`
collects current evidence and applies one eligibility policy. Valid results are excluded.
Never-accepted missing or unobserved Tasks are eligible. Unresolved acceptance withholds affected
work. Every accepted Attempt participates in retry safety: active or held work blocks retry;
terminal work requires explicit retry; unknown work additionally requires duplicate-risk
acknowledgement. Permission remains valid when uncertainty decreases to known terminal work.
Acknowledgement must belong to an explicitly retried known Task. The allocation cap bounds the reviewed batch, with excess eligible work explicitly
deferred and ineligible work explicitly excluded.

A plan records Campaign identity and revision, selected allocations and execution configuration,
retry choices, the result-probe requirement, and one integrity digest. The serialized plan is a
compact reviewed intent. Restoring it requires current coherent Campaign state, performs no remote
observation, and preserves the obligation to re-probe result-aware work. Inspection remains a
separate transient diagnostic projection.

Before each allocation, submission refreshes relevant accepted Attempts and probes selected Tasks
when a probe is bound to the handle; a result-aware plan requires one. The shared policy checks
eligibility, then an atomic transaction verifies that revision and syncs unresolved intent.
Local deterministic rendering and command validation happen before the claim. Scheduler contact
then happens outside the lock. Each Attempt stores explicit intent and outcome revisions; delayed
resolution preserves actual chronology without inventing unrecorded history.

The receipt transaction identifies the exact Attempt and records its observed outcome even if an
unrelated append or seal advanced Campaign revision. That change stops further allocations from the
stale plan. An identical receipt is idempotent; a conflicting outcome fails. Uncertain launch outcome
never proves rejection, and unresolved overlapping work blocks submission.

Submission attempts the complete reviewed batch unless an operational failure or concurrent change
stops it. Completed submission returns `SubmitResult`; operational failure raises `SubmissionError`
with the partial result and original chained cause. The result preserves confirmed receipts and
identifies unresolved and unattempted allocations. Invalid plans fail directly before submission.
An accepted receipt whose persistence failed is reported separately as observed
but not durable. `KeyboardInterrupt` and `SystemExit` propagate, leaving durable intent available
for recovery. Reconciliation and explicit accepted/not-submitted resolution preserve the same
Attempt history; no automatic retry hides ambiguity.

## Observation and diagnostics

`Campaign.inspect(scheduler=True)` projects all Attempts and current Task evidence at
one revision. The handle's optional synchronous probe runs once per Task and validates an immutable or
version-addressed canonical result. Answers are never persisted. Scheduler observation uses bounded
native calls on each Attempt's original route, with the identity, accounting-window, and retained
queue-evidence rules in ADR 0003. Incoherent scheduler evidence or concurrent Campaign mutation
rejects the observation.

The latest accepted Attempt owns current Task execution; unresolved acceptance dominates it. Result
readiness requires a sealed roster and valid results for every Task. Quiescence independently
requires requested scheduler evidence, no unresolved acceptance, and terminal evidence for every
accepted Attempt. Scheduler completion does not establish application validity, and valid results
do not establish stopped work.

`Campaign.read_log()` snapshots one accepted Attempt and derives its route, job identity, and Task
slot under the store lock, then performs a bounded binary suffix read outside the lock. Failures
raise a redacted `ObservationError` without partial bytes. Log content never enters lifecycle state
or decisions. The remote log namespace remains account-controlled and unauthenticated.

`CampaignView.to_json()` serializes the self-contained diagnostic snapshot, including task result
states, observation timestamps, and scheduler-provided diagnostic text. Serialization does not reread
Campaign state; snapshots remain exportable after append or seal. Execution configuration, task
arguments and bytes, scripts, log content, and application output contents are excluded. Scheduler
text and identifying labels may themselves contain private information, so snapshots remain private
diagnostic data with no execution authority.

The CLI mirrors explicit authoring through `create`, `append`, and `seal`; `plan` loads existing
state and reads cwd `SERVATUS.toml` unless `--config PATH` selects another file. Task JSONL accepts
optional `stdin_file`, with empty stdin when omitted. `validate`, `submit`, `status`, `logs`, `reconcile`, and
`resolve` use the same Campaign authority. Publication and Workspace ownership remain defined by
ADRs 0001, 0002, and 0004.
