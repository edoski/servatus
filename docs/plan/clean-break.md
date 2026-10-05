# Servatus 0.12 clean-break implementation spec

This is the binding contract for the 0.12 rewrite. Clean break: no compatibility shims, no
legacy aliases, no migration of 0.11 state (schema 6 is rejected; schema 7 is current). Delete
this file once the rewrite lands.

## Constraints

- Zero runtime dependencies, Python >= 3.11 (no PEP 695 syntax), pyright strict over `src` and
  `tests` (test files may opt down with a `# pyright: standard` first line), ruff, vulture.
- Tests use synthetic temp dirs only. Never contact SSH, Slurm, Apptainer, or external storage.
- Test file basenames must be unique across the whole `tests/` tree (pytest prepend import mode;
  `tests/` is on `pythonpath`, so `from support.builders import target` works).
- Every `pytest.raises` must use `match=` (or assert on the exception type precisely and a
  message), so tests fail for the right reason.
- Verification for every phase: `uv run pytest -q`, `uv run ruff check .`,
  `uv run ruff format --check .`, `uv run pyright`, `uv run vulture`.

## Package layout and ownership

```
src/servatus/
  __init__.py            entry points + __version__ (importlib.metadata)       [INTEGRATION]
  errors.py              public error hierarchy                                 [FOUNDATION]
  testing.py             FakeScheduler for user and package tests              [SCHEDULER]
  cli.py                 argparse adapter over the public API only              [INTEGRATION]
  __main__.py
  _fs.py                 pinned-descriptor POSIX primitives (shared)           [PUBLICATION]
  publication/__init__.py   publish, publish_file, Workspace, Draft, Publication [PUBLICATION]
  publication/_transaction.py  stage/commit/retire transaction                 [PUBLICATION]
  publication/_workspace.py    Workspace, child workspaces, identity           [PUBLICATION]
  campaign/__init__.py   public campaign surface                               [INTEGRATION]
  campaign/_codec.py     strict dataclass codec, canonical JSON, durations     [FOUNDATION]
  campaign/_config.py    Task, Resources, Apptainer, Target, Profile, TOML     [FOUNDATION]
  campaign/_evidence.py  types [FOUNDATION]; pure squeue/sacct parsing          [SCHEDULER]
  campaign/_results.py   Receipt, SubmitResult, ShapeCheck, LogSnapshot, ...   [FOUNDATION]
  campaign/_state.py     State, Attempt, invariant, transitions, schema 7      [FOUNDATION]
  campaign/_store.py     locked durable store with content cache               [CORE]
  campaign/_status.py    ResultState, Status types, pure projection            [CORE]
  campaign/_policy.py    Hold, Retry, eligibility decision (pure)              [CORE]
  campaign/_remote.py    Transport protocol, Ssh, Local, bounded runner        [SCHEDULER]
  campaign/_script.py    batch script, sbatch argv, log paths, names (pure)    [SCHEDULER]
  campaign/_scheduler.py Scheduler: submit/test/observe/identify/tail/cancel   [SCHEDULER]
  campaign/_plan.py      Decision, Plan, capacity, packing, build_plan (pure)  [INTEGRATION]
  campaign/_campaign.py  Campaign facade (the only imperative shell)           [INTEGRATION]
```

The old modules (`_model.py`, `_campaign.py`, `_observation.py`, `_slurm.py`, `_store.py`,
`_errors.py`, `_posix.py`, `_workspace.py`) and their tests are deleted. PUBLICATION deletes
`_posix.py`/`_workspace.py` and their tests; INTEGRATION deletes the rest.

Dependency rules: `publication` never imports `campaign`. Pure modules (`_codec`, `_config`,
`_evidence`, `_state`, `_status`, `_policy`, `_script`, `_plan`) never spawn processes, touch the
filesystem (except `Profile.load` reading TOML), or read the clock. Only `_campaign.py` composes
store, scheduler, probe, and clock. `cli.py` imports only public names. No
`reportPrivateUsage` pragmas: cross-module helpers inside private modules have no underscore.

## Foundation (frozen)

`errors.py`, `_codec.py`, `_config.py`, the type section of `_evidence.py`, `_results.py`, and
`_state.py` are written. Treat their public names and semantics as fixed. Additive changes
(new optional fields, new helpers) are allowed only when needed; list them in your report.

Key facts:
- `Task(key, args=(), *, stdin=b"", env=None)`; env names starting with `SERVATUS_` are
  reserved. Tasks pickle and deep-copy.
- `Resources(cpus, memory_mib, time_limit, gpus=0, signal_before_end=None)`, keyword-only.
  `time_limit` is a `timedelta` rounded up to whole minutes; durations accept `timedelta` or
  `[D-]H:MM:SS` text and encode canonically.
- `Target(... host: str | None = None, container: Apptainer | None = None, ...)`. `host=None`
  runs scheduler commands locally (login node). `container=None` runs each Task's absolute
  `args[0]` directly. `max_allocations_per_submit=None` means no batch cap (default).
- `Apptainer(executable, image, binds=())`, binds `SRC[:DST[:ro|rw]]`.
- `Profile(label, target, resources)`, `Profile.load(path, *, name=None)`; TOML target keys are
  the Target field names minus `container`, plus `apptainer`, `image`, `binds`.
- `JobRef(job_id, cluster=None)`; `AllocationState` with `.active` / `.terminal`;
  `SchedulerEvidence(... retained, problem)`; `StepEvidence`; `Observation(allocation, steps)`.
- `State` validates its full invariant (linear) on construction. Transitions:
  `create`, `append`, `seal`, `record_intent -> (State, Attempt)`, `record_outcome` (idempotent
  on identical repeat, `Conflict` on conflict, `NotFound` for unknown allocation).
  `encode(state) -> bytes`, `decode(bytes) -> State` (`CorruptState` on any defect).

## Public API (target)

`servatus`: `__version__`, `Campaign`, `Task`, `Resources`, `Target`, `Apptainer`, `Profile`,
`Retry`, `publish`, `publish_file`, `Workspace`, `Draft`, `Publication`, `ServatusError`.
`servatus.errors`: the hierarchy. `servatus.campaign` and `servatus.publication`: full surfaces.
`servatus.testing`: `FakeScheduler`.

```python
ResultProbe = Callable[[Sequence[Task]], Collection[str]]   # returns keys with VALID results
Connect = Callable[[Target], Transport]                      # default: servatus.campaign.connect

class Campaign:
    @classmethod
    def create(cls, path: StrPath, tasks: Iterable[Task], *, appendable=False,
               probe=None, connect=None, clock=None) -> Campaign      # Conflict if it exists
    @classmethod
    def open(cls, path, *, probe=None, connect=None, clock=None) -> Campaign   # NotFound
    @classmethod
    def ensure(cls, path, tasks, *, appendable=True, probe=None, connect=None, clock=None)
        # create (making parent dirs) or open; existing Tasks must be identical by key
        # (Conflict naming the first mismatched key); unseen Tasks are appended in order.
    id: str                                        # property
    def tasks(self) -> tuple[Task, ...]
    def append(self, tasks) -> None; def seal(self) -> None
    def status(self, *, scheduler: bool = True) -> Status
    def plan(self, profile, *, retry: Collection[str] | Retry = (), allow_duplicate_risk=(),
             only: Collection[str] | None = None, tasks_per_allocation: int | None = None) -> Plan
    def load_plan(self, data: bytes) -> Plan       # rebuilds from Decision; digest must match
    def validate(self, plan) -> tuple[ShapeCheck, ...]
    def submit(self, plan) -> SubmitResult         # SubmissionInterrupted(result=...) on stop
    def reconcile(self, allocation_id) -> Receipt
    def mark_accepted(self, allocation_id, job_id: int, *, cluster=None) -> Receipt
    def mark_not_submitted(self, allocation_id) -> None
    def cancel(self, *, tasks=None, allocations=None) -> tuple[Receipt, ...]
    def read_log(self, *, task=None, allocation=None, max_bytes=65_536) -> LogSnapshot

class Plan:   # decision + derived allocations + digest
    campaign_id, revision, profile, selected, held: Mapping[str, Hold], deferred,
    retry, duplicate_risk, tasks_per_allocation, probe_required, allocations, digest, warnings
    def to_json(self) -> bytes
    def save(self, path: StrPath) -> None          # owner-only (0600), never overwrites
```

Renames from 0.11: `SlurmTarget`->`Target`, `ResourceRequest`->`Resources`,
`SubmissionPlan`->`Plan`, `JobReceipt`->`Receipt`, `ValidationResult`->`ShapeCheck`,
`CampaignView`->`Status`, `TaskEvidence`->`TaskStatus`, `AttemptEvidence`->`AttemptStatus`,
`AllocationEvidence`->`SchedulerEvidence`, `load`->`open`, `inspect`->`status`,
`resolve`->`mark_accepted`/`mark_not_submitted`, `restore_plan`->`Campaign.load_plan`,
`plan_document`->`Plan.to_json`, `SubmissionError`->`SubmissionInterrupted`.

## Behaviour decisions by owner

### PUBLICATION (`_fs.py`, `publication/`)
Start from the reviewed prototype at `/tmp/servatus-review.vB7p3h/proto/servatus_proto/`.
- `_fs.Pin` (fd + stat entry) with verify-or-close construction; no descriptor leak windows.
  One `write_new(dir_fd, name, data, *, mode=0o600) -> os.stat_result` and an atomic
  `replace_file(dir_fd, name, data, *, mode=0o600)` (stage + fsync + rename + dir fsync) for the
  campaign store. Ownership checks use `os.geteuid()` everywhere.
- Durability: `fsync` helper uses `fcntl.F_FULLFSYNC` on macOS, `os.fsync` elsewhere.
- No-replace commit: Linux `renameat2` via libc symbol, else via `syscall(SYS_renameat2)`,
  else fallback; a missing symbol is treated like ENOSYS. Look up once (`functools.cache`).
- One stage design: every stage is a private 0700 directory. `publish_file(destination, write)`
  hands the writer `<stage>/<destination name>` (real suffix; any write strategy incl.
  temp+rename); after the writer returns, that member is pinned, synced, and committed out of
  the stage. Fixes `np.save`/`savefig` suffix bug.
- `publish(destination, build, *, retire=None, mode=None)` and `Workspace.publish(build, *,
  mode=None)`: directories are built 0700 and set to `mode` (default: `0o777 & ~umask`) just
  before commit. `publish_file(..., mode=None)` likewise for the file (default umask-derived).
- `Draft` is invalidated when the builder returns (RuntimeError on later use).
  `Draft.link(source, destination)` and new `Draft.link_tree(source_dir, destination=".")`.
- `Workspace.discard()` (exclusive, pinned removal of private work). Entering a Workspace whose
  destination already exists reclaims leftover private work under the lifecycle lock, then
  raises `DestinationExists`. `WorkspaceConflict` messages include the private path.
- Path parameters accept `str | os.PathLike[str]`; callbacks return `object`.
- `Workspace.child` uses a private constructor, no post-construction mutation; immutable
  fully-initialized levels (no Optional stat fields / asserts).
- Errors: `DestinationExists`, `UnsafeFilesystem`, `CrossDeviceError`, `UnsupportedPlatform`,
  `WorkspaceConflict`, `Busy`, `ConfigurationError` for bad caller inputs.
- Document that hard-linked results alias their source inode (in-place rewrites change them).

### SCHEDULER (`_remote.py`, `_script.py`, `_evidence.py` parsing, `_scheduler.py`, `testing.py`)
Start from `/private/var/folders/y1/h6b6vjm114v6yhtrbzr877kc0000gn/T/tmp.pVIng78Eyd/proto/remote.py`
and `/tmp/servatus-proto.ShnG0l/tests/fake_scheduler.py`.
- `Transport` protocol `run(argv, *, stdin=b"", max_stdout=...) -> Completed`.
  `Ssh(host)`: `ssh -T -o BatchMode=yes -o LogLevel=ERROR host <remote>`; remote command
  prints a per-call random marker to stdout and stderr, then `exec /usr/bin/env -i PATH=...
  LANG=C LC_ALL=C TZ=UTC argv`; output before the markers (banners, `.bashrc` noise) is
  discarded; a missing marker is `Unavailable`. `Local()`: same scrubbed env, no shell.
  `connect(target)` returns `Ssh(target.host)` or `Local()`. All spawn/timeout/overflow/exit
  failures surface as `Unavailable` (one error boundary). Bounds: 30 s deadline, 32 args,
  16 KiB command, 4 KiB per field, 1 MiB per stream; `check_command(argv)` is pure and raises
  `ConfigurationError`.
- Script: no scratch dir, no `mktemp`/`base64`/`rm`. Each step's stdin is a single-quoted
  `printf` literal piped into `srun` (printable ASCII verbatim, `\ooo` otherwise). Env passed as
  `APPTAINERENV_NAME=<shlex-quoted>` assignments (Apptainer) or `env -i ... NAME=VALUE` (direct),
  never `--env`. Inject `SERVATUS_TASK_KEY`, `SERVATUS_ALLOCATION_ID`, `SERVATUS_SLOT`,
  `SERVATUS_JOB_ID` (from `$SLURM_JOB_ID`), `SERVATUS_RESTART_COUNT` (`${SLURM_RESTART_COUNT:-0}`).
  GPU steps forward step-local `CUDA_VISIBLE_DEVICES` (fail if unset) and
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Apptainer binds work_root plus `binds`. Each step is named
  `servatus-<allocation_id>-<slot>` (`srun --job-name`). `signal_before_end` emits
  `--signal=USR1@<seconds>`. Interrupt trap kills only recorded pids (no `${!:-}`). Waits all,
  aggregates failure. Direct launcher requires absolute `args[0]` (`ConfigurationError`).
- Evidence: single state table; add `EXPEDITING` (queued); `REVOKED` is not listed (UNKNOWN,
  retained). Requeue history: only the first sacct row must fall in the submission window; later
  rows need strictly increasing Submit. squeue/sacct disagreement within one incarnation prefers
  the more advanced state (terminal over active, running over queued); irreconcilable
  contradictions become a per-attempt `problem` (UNKNOWN, retained) instead of aborting.
  Identity violations (unrelated job ids/names) still raise `EvidenceConflict`.
- Step evidence: one extra `sacct` query (no `--allocations`) maps step rows named
  `servatus-<id>-<slot>` to `StepEvidence`; failures there degrade to `None`, never abort.
- `Scheduler(transport, slurm_bin)` methods: `ping() -> str` (sbatch --version),
  `submit(argv, script) -> JobRef`, `test_only(argv, script) -> (accepted, stdout, stderr)`,
  `observe(queries) -> dict[allocation_id, Observation]`, `identify(allocation_id, intent_at)
  -> JobRef` (`ReconciliationError` unless exactly one), `tail(path, max_bytes) -> (bytes,
  truncated)`, `cancel(allocation_id, job)`. Batch <= 16 jobs per query; reused job numbers
  separately; grouped by cluster.
- `servatus.testing.FakeScheduler`: callable as `connect`, in-memory job table answering the
  exact argv issued above, with controls `start`, `finish`, `finish_step`, `requeue`, `forget`,
  `fail_next`, `lose_next_reply`, plus call counting.

### CORE (`_store.py`, `_status.py`, `_policy.py`)
Start from `/tmp/servatus-review.awHnf7/` (memo store) and
`/tmp/servatus-proto.ShnG0l/tests/test_eligibility.py`.
- Store: owner-only directory, `.lock` flock, `campaign.json`; `Store.create(path, state, *,
  parents=False)` (`Conflict` if state exists), `Store.open(path)` (`NotFound`),
  `read() -> State`, `update(change: Callable[[State], State]) -> State` (no write when the
  result is the same object). Content cache: re-decode only when file bytes differ. Integrity
  failures `UnsafeFilesystem`/`CorruptState`. Use a local atomic write for now; INTEGRATION
  switches it to `_fs.replace_file`.
- `_status`: `ResultState` (UNOBSERVED, MISSING, VALID); `TaskStatus(key, result, current_allocation_id,
  execution: AllocationState | None (step state if known), exit_code, unresolved)`;
  `AttemptStatus(allocation_id, task_keys, retry, duplicate_risk, profile_label, acceptance,
  job, intent_at, scheduler: SchedulerEvidence | None, steps)`; `Status(campaign_id, revision,
  sealed, observed_at, scheduler_observed, tasks, attempts, results_ready, quiescent)` with
  `counts() -> Mapping[str, int]` and `to_json() -> bytes` (`{"format": "servatus.status/1"}`).
  `project(state, observations, results, *, scheduler_observed, observed_at) -> Status`.
- `_policy`: `Hold` (VALID, UNRESOLVED, ACTIVE, FINISHED, UNOBSERVABLE, NOT_REQUESTED);
  `Retry` (FAILED: terminal scheduler failure/cancel of the current attempt per step/allocation;
  INCOMPLETE: any terminal accepted Task without a VALID result, requires a probe).
  `decide(state, observations, results, *, retry, duplicate_risk, only) -> Selection(selected,
  held, retry, duplicate_risk)`. Explicit retry keys raise `PlanRefused` (listing all offending
  keys) for: unknown key, VALID result, no accepted history, UNKNOWN evidence without
  acknowledgement, acknowledgement without retry. Active/unresolved/unobservable explicit
  retries are held, not raised. Bulk selectors never raise for individual Tasks.
  `observation_scope(state, retry, only) -> tuple[allocation_id, ...]`: only accepted attempts
  of Tasks that could be retried (fresh-only plans observe nothing).

### INTEGRATION (after phases above merge)
`_plan.py`, `_campaign.py`, `campaign/__init__.py`, `servatus/__init__.py`, `cli.py`, deleting
legacy modules/tests, behaviour tests over `FakeScheduler`, CLI tests.
- Plan `Decision` includes a random `nonce`; allocation ids derive from it. `submit` and
  `validate` rebuild the plan from the stored Decision and current state and compare digests
  before any claim (fixes tampered in-memory plans). Packing: balanced groups, ceilings,
  `tasks_per_allocation`, `max_allocations_per_submit`; check capacity before any observation.
- `submit`: ping transport first (no intent on broken SSH); observe once per submit; per
  allocation: revision check, `record_intent`, scheduler contact outside the lock,
  `record_outcome`. Stops raise `SubmissionInterrupted` with a complete `SubmitResult`.
- Probes are called once per operation with the relevant Tasks.
- CLI: subcommands with `set_defaults(run=...)`; exit 0 ok, 1 error, 2 usage (argparse only),
  3 submission interrupted, 75 `Unavailable`/`Busy`; `servatus: error: ...` on stderr without
  usage banner; human output by default, `--json` for machine output; `--version`.
  Commands: create, ensure, append, seal, plan (prints selected/held/deferred, `--retry KEY`,
  `--retry-failed`, `--retry-incomplete` n/a without probe so omit, `--allow-duplicate-risk`,
  `--only`, `--tasks-per-allocation`, `--output`, `--show-scripts`), validate, submit, status
  (`--offline`, `--json`), logs (`--task`/`--allocation`, `--bytes`, `--output FILE` 0600,
  refuses a TTY), reconcile, mark-accepted, mark-not-submitted, cancel, doctor (`--profile`).

### DOCS (final phase)
README as quickstart -> concepts -> recovery -> guarantees -> reference; CONTEXT.md, ADRs
(0003/0005 rewritten for transport, launcher, steps; new ADR for 0.12 scope), SECURITY.md fixes,
CONTRIBUTING, new CHANGELOG.md, CI matrix 3.11-3.14 with deeper installed-artifact smoke,
publish workflow gated on verify, tag/version check.
