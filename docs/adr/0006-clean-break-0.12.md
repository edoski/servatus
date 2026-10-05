# ADR 0006: Make a clean break in 0.12

Status: accepted

## Context

By 0.11 Servatus worked in production but had grown awkward. Names described mechanisms rather
than what users do (`SlurmTarget`, `ResourceRequest`, `inspect`, `resolve`), the error hierarchy
mirrored modules instead of the caller's next step, and several behaviours were wrong in ways
that small patches could not fix cleanly: `publish_file` broke writers that add a suffix or write a
temporary file and rename it, a tampered in-memory plan could be submitted, a broken SSH connection
still recorded intent, one contradictory `sacct` row aborted a whole observation, login-shell noise
could corrupt scheduler output, and macOS syncs did not reach stable storage. The engine mixed
decisions with process and filesystem effects, so tests monkeypatched remote calls.

Servatus has few users and no promise of stability before 1.0. Compatibility shims would keep the
old names and the old state format alive indefinitely.

## Decision

0.12 is a clean break. There are no legacy aliases, no compatibility decoders, and no migration of
0.11 state: schema 6 is rejected and schema 7 is current. Finish or abandon 0.11 campaigns with
0.11.

**Renamed.** `SlurmTarget` to `Target`, `ResourceRequest` to `Resources`, `SubmissionPlan` to
`Plan`, `JobReceipt` to `Receipt`, `ValidationResult` to `ShapeCheck`, `CampaignView` to `Status`,
`TaskEvidence` to `TaskStatus`, `AttemptEvidence` to `AttemptStatus`, `AllocationEvidence` to
`SchedulerEvidence`, `Campaign.load` to `Campaign.open`, `inspect` to `status`, `restore_plan` to
`Campaign.load_plan`, `plan_document` to `Plan.to_json`, `SubmissionError` to
`SubmissionInterrupted`. `resolve` splits into `mark_accepted` and `mark_not_submitted` (CLI
`mark-accepted`, `mark-not-submitted`). TOML resource keys drop their suffix: `cpus`, `memory_mib`,
`gpus`. The top-level `servatus` package exports only the core workflow (`Campaign`, `Task`,
`Profile`, `Target`, `Apptainer`, `Resources`, `Retry`, `publish`, `publish_file`, `Workspace`,
`Draft`, `Publication`, `ServatusError`); everything else lives in `servatus.campaign`,
`servatus.publication`, `servatus.errors`, and `servatus.testing`.

**Errors.** The hierarchy in `servatus.errors` is grouped by what the caller can do:
`ConfigurationError` (fix the input), `NotFound`, `Conflict` (with `StalePlan`,
`DestinationExists`, `WorkspaceConflict`), `PlanRefused` (decide explicitly), `Busy` (retry soon),
`Unavailable` (retry later), `IntegrityError` (with `CorruptState`, `UnsafeFilesystem`,
`EvidenceConflict`; investigate), `ReconciliationError`, and `SubmissionInterrupted`. The CLI maps
them to exit codes 0, 1, 2, 3, and 75.

**Reshaped.** The campaign engine is a functional core with one imperative shell (ADR 0005).
Plans hold every unselected Task with one `Hold` reason; Tasks with accepted work are held
`SUBMITTED` unless a retry is requested, so plans of fresh work never consult the scheduler. The
CLI prints human-readable output, with `--json` on every command except `logs`; `plan --output` is
optional.
Scheduler access goes through one `Transport` seam with `Ssh` and `Local` implementations, and Tasks
start through an Apptainer or a direct launcher (ADR 0003). Publication uses one stage-directory
design for files and directories (ADR 0002).

**Added.** Local execution on a login node (omit `host`); the direct launcher; per-Task step
evidence and exit codes; `signal_before_end`; injected `SERVATUS_*` variables; `Campaign.ensure`;
`Retry` selectors and `only=`; `cancel`; `read_log` by Task; `Plan.save`; public `capacity`;
`status --offline`; `doctor`; `--version`; `servatus.testing.FakeScheduler`; `Draft.link_tree`;
`Workspace.discard`; `mode=` on publication.

**Removed.** Batch scratch payload files, `--env` forwarding, the 0.11 error classes, and the
default one-allocation submission cap (`max_allocations_per_submit` now defaults to no cap).

## Consequences

Users rewrite imports, configuration keys, and CLI invocations once, guided by the changelog.
Existing 0.11 campaign directories cannot be opened. The scope stays as narrow as before: one
native Slurm lane, single-node allocations, no arrays, no automatic retry, and no scheduler
plugins. One Task may use several GPUs on its node, so single-node multi-GPU programs such as
`torchrun --standalone` are supported. The 0.11 production acceptance recorded in ADR 0003 does not
cover the new transport and launcher; a new live gate is required before claiming it does.
