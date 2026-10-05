# Changelog

All notable changes to Servatus are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (before 1.0, minor versions may break compatibility).

## [Unreleased]

## [0.12.0]

A clean break: no compatibility aliases and no migration of earlier campaign state. See
[ADR 0006](https://github.com/edoski/servatus/blob/main/docs/adr/0006-clean-break-0.12.md).

### Breaking

- Campaign state uses schema 7. Schema 6 state from 0.11 is rejected with `CorruptState`; finish
  or abandon 0.11 campaigns with 0.11. Plan files are now tagged `servatus.plan/1`, so 0.11 plan
  files are rejected too.
- The top-level `servatus` package exports only `Campaign`, `Task`, `Profile`, `Target`,
  `Apptainer`, `Resources`, `Retry`, `publish`, `publish_file`, `Workspace`, `Draft`,
  `Publication`, and `ServatusError`. Everything else is imported from `servatus.campaign`,
  `servatus.publication`, `servatus.errors`, or `servatus.testing`.
- Renamed: `SlurmTarget` → `Target`, `ResourceRequest` → `Resources`, `SubmissionPlan` → `Plan`,
  `JobReceipt` → `Receipt`, `ValidationResult` → `ShapeCheck`, `CampaignView` → `Status`,
  `TaskEvidence` → `TaskStatus`, `AttemptEvidence` → `AttemptStatus`, `AllocationEvidence` →
  `SchedulerEvidence`, `SubmissionError` → `SubmissionInterrupted`.
- Renamed methods: `Campaign.load` → `Campaign.open`, `Campaign.inspect` → `Campaign.status`,
  `restore_plan` → `Campaign.load_plan`, `plan_document` → `Plan.to_json`. `resolve` is replaced
  by `Campaign.mark_accepted` and `Campaign.mark_not_submitted`.
- CLI: `resolve` is replaced by `mark-accepted` and `mark-not-submitted`; `logs` takes `--task`,
  `--allocation`, or both instead of a positional allocation; usage errors exit 2, interrupted
  submissions exit 3, and an unavailable cluster or busy campaign exits 75.
- CLI output is human-readable; pass `--json` (every command except `logs`) for machine output.
  `status --json` prints a canonical `servatus.status/1` document with `counts`.
- `servatus plan` prints the plan and saves it only with `--output`, which is no longer required.
  `servatus validate` exits 1 when Slurm rejects any allocation shape.
- Errors are regrouped in `servatus.errors` by what the caller can do: `ConfigurationError` (with
  `CrossDeviceError` and `UnsupportedPlatform`), `NotFound`, `Conflict` (`StalePlan`,
  `DestinationExists`, `WorkspaceConflict`), `PlanRefused`, `Busy`, `Unavailable`, `IntegrityError`
  (`CorruptState`, `UnsafeFilesystem`, `EvidenceConflict`), `ReconciliationError`, and
  `SubmissionInterrupted`. The 0.11 classes (`CampaignError`, `PlanError`, `ObservationError`,
  `PublicationError`, `UnsafePublication`, `CrossDevicePublication`, `WorkConflict`,
  `WorkspaceBusy`, `TaskConflict`) are gone.
- `Task` takes `stdin` and `env` as keyword arguments: `Task(key, args, *, stdin=b"", env=None)`.
  Environment names starting with `SERVATUS_` are reserved.
- `Resources` fields and TOML keys are `cpus`, `memory_mib`, `gpus`, and `time_limit` (previously
  `cpus_per_task`, `memory_mib_per_task`, `gpus_per_task`). `time_limit` is stored rounded up to
  whole minutes.
- The Apptainer launcher is a separate `Apptainer(executable=..., image=..., binds=...)` value on
  `Target.container`. TOML keeps the `apptainer` and `image` keys, now optional.
- `max_allocations_per_submit` defaults to no cap instead of one allocation per plan.
- Plans report held Tasks as a mapping of key to `Hold` reason and deferred Tasks as `deferred`,
  replacing `excluded_task_keys` and `deferred_task_keys`. A Task with accepted work is held
  `SUBMITTED` unless a retry is requested for it, and its scheduler state is not consulted.
- The result probe receives a sequence of Tasks and returns the keys with valid results; it is
  called once per operation instead of once per Task.
- `publish_file` writers receive `<private stage>/<destination name>` instead of a pre-created
  empty file, and may create it by any means. The file's mode is set just before commit (default
  `0o666` minus the umask), overriding any mode the writer chose.
- Invalid publication input (an unsafe draft path, a missing destination parent, a non-sibling
  `retire`, a bad `mode`, a non-bytes identity) raises `ConfigurationError` instead of
  `UnsafePublication`.

### Added

- `host` is optional: omit it to run Slurm commands locally on a login node.
- Direct launcher: omit the Apptainer settings to run each Task's absolute `args[0]` under a clean
  environment.
- Apptainer `binds` (`SRC[:DST[:ro|rw]]`).
- Per-Task step evidence: each Task runs in a step named `servatus-<allocation>-<slot>`, and status
  reports its own state and exit code.
- `Resources.signal_before_end` sends `SIGUSR1` to each Task before its time limit.
- Workers receive `SERVATUS_TASK_KEY`, `SERVATUS_ALLOCATION_ID`, `SERVATUS_SLOT`,
  `SERVATUS_JOB_ID`, and `SERVATUS_RESTART_COUNT`.
- `Campaign.ensure` (CLI `ensure`, created appendable unless `--sealed`) creates a campaign or
  appends unseen Tasks idempotently.
- `Retry.FAILED` and `Retry.INCOMPLETE` bulk retry selectors (CLI `--retry-failed`), `only=`
  (CLI `--only`), and explicit `Hold` reasons. `Retry.INCOMPLETE` requires a result probe
  (`ConfigurationError` when planning without one), and duplicate risk can be acknowledged only
  for explicitly retried keys.
- `Campaign.cancel` (CLI `cancel`) stops every accepted allocation of the named Tasks, or the named
  allocations, unless Slurm already reports them finished.
- `servatus.campaign.capacity` reports how many Tasks fit one allocation.
- `Campaign.read_log` by Task, allocation, or both; CLI `logs --output FILE` writes owner-only files and
  refuses to print to a terminal.
- `Plan.save` writes owner-only plan files without overwriting.
- `servatus status --offline`, `servatus doctor`, and `servatus --version`.
- `servatus.testing.FakeScheduler`, an in-memory Slurm for testing launchers and recovery code.
- `Draft.link_tree` hard-links a whole source tree into a draft.
- `Workspace.discard` removes private work without publishing.
- `mode=` on `publish`, `publish_file`, and `Workspace.publish` sets the published permissions.
- Path parameters accept any `os.PathLike[str]`.

### Changed

- Campaign state stores each distinct Profile once instead of once per Attempt.
- Submission observes the scheduler once per submission instead of once per allocation, and
  planning observes only Attempts whose Tasks could be retried.
- The store caches decoded state and decodes again only when the file changes.
- Batch scripts pass stdin as `printf` literals and environment as `APPTAINERENV_*` assignments;
  they create no scratch files.
- Directories are built owner-only and every published directory receives its final mode just
  before commit.
- Scheduler replies may hold up to 4096 lines (previously 128).
- `REVOKED` is `UNKNOWN` and retained instead of `CANCELLED`.

### Fixed

- `publish_file` works with writers that add a suffix or write a temporary file and rename it
  (`np.save`, `savefig`).
- `submit` and `validate` rebuild the plan from its recorded decision and refuse a tampered
  (`ConfigurationError`) or stale (`StalePlan`) in-memory plan before claiming any work.
- Submission checks the scheduler connection and rechecks eligibility first, so a broken SSH
  connection no longer records unresolved intent; these failures raise ordinary errors rather than
  `SubmissionInterrupted`.
- Login banners and shell-startup output on the cluster can no longer corrupt scheduler output.
- One contradictory allocation no longer aborts the whole observation; it is reported as a
  per-allocation `problem` and only its Tasks are withheld. That includes a queried job number now
  held by a foreign job.
- Requeued jobs whose later incarnations fall outside the original submission window are tracked
  correctly: only the first accounting row must fall inside it.
- `EXPEDITING` jobs are recognized as queued.
- The batch interrupt trap signals only the steps it started.
- Environment values containing commas or equals signs reach Apptainer intact.
- Durability on macOS uses `F_FULLFSYNC`.
- No-replace commits work on Linux systems whose libc lacks the `renameat2` wrapper.
- Entering a Workspace whose destination already exists reclaims leftover private work.
- Filesystem pins no longer leak descriptors when verification fails.
- A `Draft` can no longer be used after its builder returns.

## Earlier versions

Versions before 0.12.0 had no changelog. See the git history for their changes.

[Unreleased]: https://github.com/edoski/servatus/compare/v0.12.0...HEAD
[0.12.0]: https://github.com/edoski/servatus/releases/tag/v0.12.0
