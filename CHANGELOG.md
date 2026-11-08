# Changelog

All notable changes to Servatus are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (before 1.0, minor versions may break compatibility).

## [Unreleased]

A long Task can now continue across allocations under a lifetime run-time budget, with no
controller running while it does: `plan --budget` queues a chain of allocations at submission, and
each link hands its Tasks on to the next through a marker it writes after their steps returned (see
[ADR 0008](https://github.com/edoski/servatus/blob/main/docs/adr/0008-chains.md)). `status` shows
every Task's run time, summed over its allocations from run-marker mtimes. A **lease** holds one
allocation, GPUs included, for its whole time limit, and Tasks start, stop, and are watched inside
it from the operator's machine as steps: `plan --hold N` and `submit` make one, `start` starts the
Tasks of a file in it, and `cancel`, `status`, `logs`, and `reconcile` work as for any Attempt (see
[ADR 0009](https://github.com/edoski/servatus/blob/main/docs/adr/0009-leases.md)). Plans without a
budget or a lease render the batch scripts and `sbatch` requests 0.13 did, plus
`SERVATUS_JOB_END_TIME` in every step.

**Upgrading from 0.13.** Schema 8 campaign state and `servatus.plan/2` plan files are rejected, so
finish or abandon 0.13 campaigns with 0.13 first, then create new campaigns and plans. Published
results and Workspaces are unaffected, so a Task rerun in a new campaign still resumes its private
work. Read `servatus.status/4` from `status --json` (timed observations, `run_time`, `link`, and
each Attempt's `slots` and `within`), expect exit 75 to read as `STOPPED (yielded)`, make sure the
login host has GNU `find` (run `servatus doctor`), and add `signal_before_end` to a profile before
planning with `--budget` or holding a lease. `plan` and `doctor` still read `./SERVATUS.toml` unless
`--config` is given.

### Breaking

- Campaign state uses schema 10: every Attempt records its `link`, its place in a chain of
  allocations (`chain`, the allocation id of link 1, its 1-based `index`, the chain's `count`, and
  the `budget` the chain was planned with), and its `slots` and `within`, which make it a lease or
  a step. A plain allocation is link 1 of 1. The invariant requires every later link to follow the
  accepted previous link of its chain with the same Tasks and slots, and a step to fit a lease
  accepted before it, on the lease's Profile and job. Schema 8 is rejected; there is no migration.
- `status --json` prints `servatus.status/4`. Each Task gains `run_time` (`{"total", "complete"}`,
  `null` offline) and `link`, and `detail` is one of `never started`, `time limit`, `yielded`, or
  `budget spent`. Each Attempt gains `link`, `slots`, and `within`, and its `observation` replaces
  `started`, `stopped`, and `exit_codes` with `started_at`, `stopped_at`, `limit_at`, `ends`
  (`{"exit_code", "at"}` or `null` per slot), `handed`, and `step`. `Observation.execution`
  returns the execution, the exit code, and a `StopReason`.
- A step that exits 75 reads as `STOPPED (yielded)` with exit code 75 in every campaign, not
  `FAILED`. `Retry.FAILED` selects both, as before.
- Run markers are listed with `find -printf '%T@ %p\n'`, which needs GNU find on the login host.
  `servatus doctor` gains a `find -printf` check.
- Saved plans are `servatus.plan/4`: the Decision gains `continuation`, `spent` (each selected
  Task's measured run time, with a continuation or for steps), `extensions`, `slots`, and
  `within`. `PlannedAllocation` gains `link`, `slots`, and `within`, and its `argv` of a later link
  lacks the dependency that `submit` adds.
- `Campaign.plan`'s `profile` is optional, so a step plan can take its Profile from its lease.
- `cancel` stops allocations newest first, and cancelling a chain link also cancels every later
  link of its chain.
- `logs --task`, status, and `--retry-failed` follow a Task's current allocation: its latest
  accepted Attempt that was handed the Task.

### Added

- Run-time accounting: `status` shows each Task's `RUN TIME`, summed over every allocation that ran
  it from run-marker mtimes, and a `LINK` column. `RunTime`, `StepEnd`, `StopReason`, and `Link` are
  public.
- `FakeScheduler` stamps file mtimes from a clock (`clock=`, or an internal one moved by
  `advance`), reports them through `find -printf`, and exposes `mtime(path)` and `now()`.
- Budgeted continuation: `plan --budget DURATION [--max-links K]`
  (`Campaign.plan(..., continuation=Continuation(budget=..., max_links=...))`) gives each selected
  Task a lifetime run-time budget and queues a chain of links to spend it, full `time_limit` links
  and one shorter last link. Each later link depends on the earlier ones (`afternotok`,
  `--kill-on-invalid-dep=yes`) and runs only the Tasks handed to it. A link that hands off gets the
  pre-limit signal in its batch shell (`--signal=B:USR1@N`), writes a limit marker, and relays it to
  its steps. Steps learn `SERVATUS_BUDGET`, `SERVATUS_RUN_TIME`, `SERVATUS_LINK`, and
  `SERVATUS_LINKS`; exit 75 (`YIELD_EXIT`) continues, exit 0 is done. `validate` checks later-link
  shapes with `--kill-on-invalid-dep=yes`.
- Extension: `--retry KEY --budget D` continues a `SUCCEEDED` Task, or one whose probe reports
  `VALID`, with a larger lifetime budget; `Plan.extensions` and the plan output name each. A new
  hold, `SPENT`, keeps a Task whose budget is spent from rerunning. `Continuation`, `Extension`,
  `MAX_LINKS`, and `YIELD_EXIT` are public.
- `FakeScheduler` emulates chains: `afternotok` dependencies with `--kill-on-invalid-dep=yes`, gated
  links, `signal_limit` for the pre-limit signal, and handoffs by the same rule the shell
  implements. `FakeJob` gains `dependencies`.
- Every step learns its job's deadline as `SERVATUS_JOB_END_TIME` (UNIX seconds), passed from
  Slurm's `SLURM_JOB_END_TIME`, which Slurm 23.02 and later set in every batch job. Like
  `SLURM_JOB_ID`, the batch script requires it and exits before any run marker without it.
- `plan` prints the most a plan requests, `requests at most N CPU-hours, N GPU-hours, N GiB-hours`
  over every allocation and chain link at its time limit (GPU-hours only when nonzero). `plan
  --json` carries it as `totals`, and the library as `Plan.totals`, a public `Totals`.
- Leases: `plan --hold N [--max-links K]` (`Campaign.hold(profile, slots=N)`) plans one lease of N
  slots of the profile's per-Task resources, or a chain of K, and `submit` sends it as one job that
  gets the pre-limit signal in its batch shell. `plan --lease ID [--slots K]`
  (`Campaign.plan(lease=ID, slots=K)`) plans each selected Task as a step of K slots in the lease's
  current link, and `submit` writes its order into the lease's owner-only order directory instead
  of calling `sbatch`. `servatus start STATE TASKS.jsonl --lease ID` registers the file's new Tasks
  and starts its keys in one command. The lease's POSIX `sh` batch script starts each step only
  after its no-replace claim on the order won, the same claim `cancel` and `reconcile` contend
  for; the step writes its own start marker once Slurm gave it room, so a waiting step reads
  `QUEUED` and accrues no run time. `cancel --task KEY` withdraws a step the lease has not taken,
  or has the lease stop it with `SIGUSR1`, then `SIGTERM` after `signal_before_end`; after the
  pre-limit signal the lease takes no more orders. A step stops being live as soon as it ended or
  was withdrawn, so its Task can be started again in the same lease. `reconcile` resolves an
  unresolved step at once. A new run marker, `<step>.<job>.void`, records a withdrawn step. Steps
  get the Campaign's step environment, `SERVATUS_JOB_END_TIME` included, and request nothing in
  `Plan.totals`. `Kind` is public, and `FakeScheduler` emulates leases with
  `poll`, `begin_step`, `end_step`, and `honor_stops`.

## [0.13.0] - 2026-11-06

Servatus no longer needs Slurm accounting, and a new cluster is a new profile, not new code. It
observes work from `squeue` and empty run markers each batch script writes (see
[ADR 0007](https://github.com/edoski/servatus/blob/main/docs/adr/0007-run-markers.md)),
`servatus doctor` checks what a profile needs from its cluster, and the README configures both
DISI clusters. Like 0.12, this is a clean break with no compatibility layer.

**Upgrading from 0.12.** This is a migration, not a version bump. Schema 7 campaign state and
`servatus.plan/1` plan files are rejected, so finish or abandon 0.12 campaigns with 0.12 first, then
create new campaigns and plans under 0.13. Published results and Workspaces are unaffected, so a
Task rerun in a new campaign still resumes its private work. Replace `mark-accepted`,
`mark-not-submitted`, and `reconcile STATE ALLOCATION` with `reconcile STATE`, stop passing
`--allow-duplicate-risk` and `retire=`, read `servatus.status/2` from `status --json`, and run
`servatus doctor` on every profile.

### Breaking

- Campaign state uses schema 8, saved plans are tagged `servatus.plan/2`, and `status --json` prints
  `servatus.status/2`. Schema 8 stores only what history cannot derive: an Attempt keeps its job and
  outcome revision (no `acceptance` or `retry`), and each roster entry nests its Task under `"task"`.
  A saved plan's Decision has no `retry`; `Plan.retry` derives it, and `servatus plan --json` shows
  it.
- Observation reads `squeue --me --all --states=all` and empty run markers that each batch script
  writes to `log_root` (`<allocation>.<job>.start`, `<allocation>.<job>.<slot>.<exit code>.end`,
  `<allocation>.<job>.stop`), never `sacct`. A step that returned shows `SUCCEEDED` or `FAILED` as
  soon as its marker is visible. `STOPPED`, `LOST`, and every retry wait until Slurm purges the job
  (`MinJobAge`), so a cancelled Task reads `ENDING` until then and `--retry-failed` picks failed
  work up a few minutes after it ends.
- Executions are `QUEUED`, `RUNNING`, `ENDING` (Slurm reports the job ended; the outcome is
  final once Slurm forgets it), `SUCCEEDED`, `FAILED`, `STOPPED` (interrupted or never started),
  and `LOST` (vanished without recording an outcome). `AllocationState`, `UNKNOWN`,
  `CANCELLED`, `SchedulerEvidence`, `StepEvidence`, and `JobRef` are replaced by `Execution` and
  `Observation`; job numbers are plain integers and `TaskStatus.exit_code` is the step's exit code.
- `TaskStatus.execution` is the one field for a Task's state: its observed `Execution`, or
  `AcceptanceState.UNRESOLVED`, `ACCEPTED` (scheduler not observed), or `NOT_SUBMITTED`.
  `TaskStatus.unresolved` is removed, and the `status` table shows these names instead of `-`.
- A `servatus.status/2` document differs from `/1`: `attempts[].scheduler`, `steps`, and
  `duplicate_risk` become `attempts[].observation` (`slurm_state`, `started`, `stopped`,
  `exit_codes`); `attempts[].job` and `tasks[].exit_code` are integers; `tasks[].execution` is never
  null; and `counts` replaces `unsubmitted`, `cancelled`, and `unknown` with `not_submitted`,
  `ending`, `stopped`, and `lost`.
- The `UNOBSERVABLE` hold, duplicate-risk acknowledgement (`allow_duplicate_risk`,
  `--allow-duplicate-risk`, `Decision.duplicate_risk`), federation clusters (`--cluster`), and
  `SERVATUS_RESTART_COUNT` are removed. `ACTIVE` means Slurm still lists earlier work, finished but
  not yet purged included. `Retry.FAILED` selects `FAILED` and `STOPPED` work, never `LOST`; an
  explicit `--retry KEY` refuses live work and accepts `LOST`.
- `Campaign.reconcile()` (`servatus reconcile STATE`) takes no allocation and resolves every
  unresolved allocation from a queue row, then a run marker, then records it as not submitted once
  its intent is 15 minutes old. It returns `ReconcileResult`; the CLI exits 75 while some wait.
  `mark_accepted`, `mark_not_submitted`, their CLI commands, and `ReconciliationError` are removed.
- Jobs are submitted with `--no-requeue` and without `--comment`; `sbatch --parsable` must print a
  bare job number. `cancel` runs `scancel --me --name=<identity>` without a job number.
- `servatus doctor` checks what a profile needs from its cluster and reports each check as `pass`,
  `fail`, or `unavailable`: Slurm 23.02 or later, `squeue` under the scrubbed environment,
  `MinJobAge` of at least 120 s, `proctrack/cgroup`, writable `work_root` and `log_root`, and a
  one-Task `sbatch --test-only`. It exits 1 if a check fails and 75 if one went unanswered. Its
  `--json` document drops `sbatch_version` for `checks` and `verdict`.
- `servatus.campaign.ping` is replaced by `servatus.campaign.diagnose(profile)`, which returns a
  `Diagnosis` of `Check`s with a `Verdict`.
- `servatus.testing.FakeScheduler` models the new protocol: queue rows that settle until `purge`,
  run markers in an in-memory `log_root`, `scancel` by name, and `find`. Its controls are `start`,
  `finish_step`, `finish`, `interrupt`, `lose`, `purge`, `write_file`, and `remove_file`.
- `publish` no longer takes `retire=`. No known consumer used it; a `Workspace` covers private work
  beside the destination, with identity binding and residue reclaim.
- `CrossDeviceError` and `UnsupportedPlatform` are gone. Those failures raise `ConfigurationError`.

### Added

- `constraint` accepts Slurm feature expressions (`a100|h100`, `intel&ib`), and `gpu_gres` accepts
  any count-free GRES (`shard`, `gpu:a100_80GB`).
- A README section with a tested `SERVATUS.toml` for each DISI cluster and the steps to onboard a
  student on `giano`.
- `Status.warnings`: a job Slurm lists for an allocation recorded as not submitted holds its
  Tasks `ACTIVE`, keeps the campaign non-quiescent, and is named until it leaves the queue.
  `cancel --allocation` can stop it.
- `TaskStatus.detail` (`"never started"` for a stopped job that started no Task).
- `scripts/cluster_e2e.py`, a live end-to-end check against a real cluster, never run by CI. It can
  also run a one-GPU Task, an Apptainer Task, and a one-GPU Apptainer Task.

### Changed

- `args` and `env` errors from a Task file now come from `Task`, so they read the same as in Python.
- A hard-link failure that is not a substitution uses the general filesystem wording (for example
  `crosses a filesystem boundary`).

### Fixed

- A draft-path collision in `Draft.link` or `Draft.link_tree` raises `ConfigurationError` instead of
  `DestinationExists`. A worker that reads `DestinationExists` as "already published" could exit 0
  with nothing published when its builder linked two files to one path.
- A status poll over SSH could fail at the 30 s deadline when one TCP connection never opened, which
  happened about once in 100 connections from a laptop on a phone hotspot. `ssh` now gives each
  connection three attempts of 8 s. Only the connection is retried, never a command, and a stalled
  banner exchange still fails, after 8 s. The deadline error names the command, and `ssh`'s own
  failure (exit 255) reports its message.

## [0.12.1] - 2026-10-05

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

[Unreleased]: https://github.com/edoski/servatus/compare/v0.13.0...HEAD
[0.13.0]: https://github.com/edoski/servatus/releases/tag/v0.13.0
[0.12.1]: https://github.com/edoski/servatus/releases/tag/v0.12.1
[0.12.0]: https://github.com/edoski/servatus/releases/tag/v0.12.0
