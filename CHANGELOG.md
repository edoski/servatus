# Changelog

All notable changes to Servatus are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (before 1.0, minor versions may break compatibility).

## [Unreleased]

### Fixed

- Batch scripts export `PATH=/usr/bin:/bin`. Under `--export=NIL` they had none, so Apptainer's
  `--nv` bound no host NVIDIA files: GPU Tasks could not reach `nvidia-smi` or an MPS daemon,
  and CUDA used whatever `libcuda` the image carried.

## [0.12.0] - 2026-10-05

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
  `restore_plan` → `Campaign.load_plan`, `plan_document` → `Plan.to_json` (canonical bytes without
  a trailing newline). `resolve` is replaced by `Campaign.mark_accepted` and
  `Campaign.mark_not_submitted`.
- The reviewed fields of a plan (`campaign_id`, `revision`, `profile`, `selected`, `held`,
  `deferred`, `retry`, `duplicate_risk`, `tasks_per_allocation`, `probe_required`) live on
  `Plan.decision`; `Plan` itself keeps only `decision`, `allocations`, `digest`, `warnings`,
  `to_json`, and `save`.
- CLI: `resolve` is replaced by `mark-accepted` and `mark-not-submitted`; `logs` takes `--task`,
  `--allocation`, or both instead of a positional allocation; usage errors exit 2, interrupted
  submissions exit 3, an unavailable cluster or busy campaign exits 75, and Ctrl-C exits 130.
- CLI output is human-readable; pass `--json` (every command except `logs`) for machine output.
  `status --json` prints a canonical `servatus.status/1` document with `counts`. JSON receipts
  nest the job as `"job": {"job_id", "cluster"}`.
- `mark_not_submitted` (CLI `mark-not-submitted`) records an outcome only after the allocation's
  original target proves no job carries its identity, so that target must be reachable.
- Apptainer steps run `apptainer exec` instead of `apptainer run`: the Task's `args` run directly
  whatever the image's runscript does, and an Apptainer Task with empty `args` is refused at
  planning.
- `Target` rejects `%` in `log_root` (Slurm expands it) and `,` or `:` in `work_root` (it is
  bound into containers); `Apptainer` rejects `:` in `image` (Apptainer reads it as a URI).
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
- Plans report held Tasks as a mapping of key to `Hold` reason (`plan.decision.held`) and deferred
  Tasks as `plan.decision.deferred`, replacing `excluded_task_keys` and `deferred_task_keys`. A
  Task with accepted work is held `SUBMITTED` unless a retry is requested for it, and its
  scheduler state is not consulted.
- The result probe receives a sequence of Tasks and returns the keys with valid results; it is
  called once per operation instead of once per Task.
- `publish_file` writers receive `<private stage>/<destination name>` instead of a pre-created
  empty file, and may create it by any means. The file's mode is set just before commit (default
  `0o666` minus the umask), overriding any mode the writer chose.
- Invalid publication input (an unsafe draft path, a missing destination parent, a non-sibling
  `retire`, a bad `mode`, a non-bytes identity) raises `ConfigurationError` instead of
  `UnsafePublication`.
- Filesystem failures in publication, Workspaces, and the campaign store never escape as a raw
  `OSError`. A missing, unwritable, or overlong path, a missing link or `retire` source, a writer
  that creates nothing, and a file its owner cannot read raise `ConfigurationError` (`NotFound`
  for missing campaign state); no space, quota, I/O errors, and descriptor exhaustion raise
  `Unavailable`. `UnsafeFilesystem` now means only substitution, wrong type, or wrong ownership.
  Exceptions raised by builders and writers propagate unchanged.

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
- `servatus.campaign.capacity` reports how many Tasks fit one allocation,
  `servatus.campaign.ping(target, *, connect=None)` returns the `sbatch --version` text (as
  `servatus doctor` does), and `servatus.campaign.to_document(value)` turns results, receipts,
  shape checks, and planned allocations into the JSON-compatible data the CLI prints.
- `Campaign.read_log` by Task, allocation, or both; CLI `logs --output FILE` writes owner-only files and
  refuses to print to a terminal.
- `Plan.save` writes owner-only plan files without overwriting.
- `servatus status --offline`, `servatus doctor`, and `servatus --version`.
- `servatus.testing.FakeScheduler`, an in-memory Slurm for testing launchers and recovery code.
  Its controls and `count` name commands by basename, except the `sbatch --version` check, whose
  key is `"ping"` (so `count("sbatch")` counts only submissions and shape checks).
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
- A Task's execution is its own step when Slurm reports it. Allocation evidence that is `UNKNOWN`
  or carries a `problem` makes all its Tasks `UNKNOWN` whatever their steps say, and a failed or
  cancelled allocation of several Tasks makes a Task without step evidence `UNKNOWN`: such Tasks
  are held `UNOBSERVABLE` by `Retry.FAILED` and need a duplicate-risk acknowledgement to retry
  explicitly. Exit codes are reported only for terminal executions (`-` in human output).
- A failed step query leaves an allocation's `steps` empty (step evidence unavailable) instead of
  one `None` per slot.
- `mark_accepted` can correct an allocation recorded as not submitted, but only when its original
  target proves exactly that job and no later allocation already includes its Tasks.
- `cancel` tries every selected allocation; if any `scancel` fails it raises `Unavailable` naming
  the cancelled and the failed allocations.
- Workspace containers are removed `work` first and `.identity` last. The same identity finishes
  an interrupted removal on its next entry; a container holding work but no identity is never
  adopted (`WorkspaceConflict` naming the private path).
- Human CLI output escapes every non-printable character (scripts printed by `--show-scripts`
  stay verbatim); `servatus doctor` failures include Slurm's first stderr line.
- The ssh client also receives `KRB5CCNAME` and `KRB5_CONFIG`, so Kerberos (GSSAPI) logins work.

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
- Publication and Workspaces work on filesystems that refuse `flock` on a directory (NFS: `EBADF`,
  `ENOLCK`, `EOPNOTSUPP`, `EINVAL`) by locking an owner-only `.servatus.lock` file in that
  directory instead. The file is never removed and cannot be shared between users.
- Cleanup removes private directories that lack owner read or search permission.
- `Draft.link_tree` refuses a source tree that contains the draft it links into
  (`ConfigurationError`).
- A receipt that became durable although its commit reported failure is reported as a receipt,
  not as unresolved work.
- A ControlPersist master holding the ssh client's stderr open no longer stalls every scheduler
  call until the deadline: the runner stops waiting for end-of-file once the client exits.
- The CLI never prints a traceback: an unexpected error is one `servatus: error: unexpected
  <type>: <message>` line with exit 1. `Profile.load` with a NUL in the path raises
  `ConfigurationError`.
- Task JSONL files are split on `\n` only, so strings may contain U+2028 and other Unicode line
  separators.

## Earlier versions

Versions before 0.12.0 had no changelog. See the git history for their changes.

[Unreleased]: https://github.com/edoski/servatus/compare/v0.12.0...HEAD
[0.12.0]: https://github.com/edoski/servatus/releases/tag/v0.12.0
