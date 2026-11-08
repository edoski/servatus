# Servatus context

Servatus uses a small generic vocabulary. Names in `code` are the public Python names.

## Campaigns

- **Task (`Task`):** one stable opaque key, argument vector, environment mapping, and stdin bytes.
  Environment names starting with `SERVATUS_` are reserved.
- **Campaign (`Campaign`):** one directory holding an ordered Task roster and its durable Attempt
  history. `create` authors a roster (sealed unless `appendable=True`), `open` reopens it, `ensure`
  creates or extends it idempotently, `append` registers only new Tasks, and `seal` ends authoring
  irreversibly.
- **Revision:** a counter advanced by every durable mutation. Plans bind to one revision.
- **Resources (`Resources`):** one homogeneous per-Task CPU, MiB, GPU (units of `gpu_gres`),
  wall-time, and optional `signal_before_end` request for one plan.
- **Target (`Target`):** one concrete Slurm route: optional SSH host, absolute Slurm directory,
  work and log roots, partitions, site options, optional launcher, and conservative ceilings. A
  second cluster is a second profile with its own Target. Every Attempt keeps its original Target
  for observation, recovery, logs, and cancellation.
- **Launcher:** how a step starts a Task. The **Apptainer launcher** (`Apptainer`) runs the Task's
  `args` with `apptainer exec` in one immutable image with a clean environment (the image's
  runscript is never used; `args` must be nonempty); the **direct launcher** (no container) runs
  the Task's absolute `args[0]` under `env -i`.
- **Profile (`Profile`):** one nonbinding label plus a complete Target and Resources, loaded from
  `SERVATUS.toml` with document-level defaults overridden per key.
- **Transport (`Transport`):** how scheduler commands reach Slurm. `connect(target)` returns an SSH
  transport (OpenSSH in batch mode, retrying only the connection, output fenced by per-call markers)
  when the Target has a host, and a local one otherwise. Both run with a scrubbed environment and
  fixed bounds and return a `Completed`; every failure surfaces as `Unavailable`. Any callable from
  Target to Transport (for example `servatus.testing.FakeScheduler`) can replace `connect`.
- **Diagnosis (`Diagnosis`):** what `diagnose(profile)` and `servatus doctor` found, read-only: one
  `Check` with a `Verdict` (`pass`, `fail` when a command ran and said no, `unavailable` when no
  answer came) for each capability Servatus needs from a profile's cluster: Slurm 23.02 or later,
  `squeue` under the scrubbed environment, `MinJobAge` of at least 120 s, `proctrack/cgroup`,
  writable roots, `find -printf` on `log_root`, and a one-Task `sbatch --test-only`.
- **Allocation:** one single-node Slurm job named `servatus-<allocation_id>`, submitted with
  `--no-requeue`, running its Tasks as concurrent exact steps. Each step is named
  `servatus-<allocation_id>-<slot>`.
- **Run marker:** a file the batch script creates in `log_root`, read by name and mtime:
  `<allocation_id>.<job>.start` before any step, `<allocation_id>.<job>.<slot>.<exit code>.end`
  when that slot's step returns, and `<allocation_id>.<job>.stop` when the script is interrupted.
  A chain link that hands off, and a lease, also write `<allocation_id>.<job>.limit` when their batch
  shell gets the pre-limit signal. A step's markers carry its lease's job (see Step).
- **Chain:** the allocations, called links, that one budgeted plan queues for one packed group of
  Tasks. `submit` sends them all at once, in order; each later link depends on every earlier one
  (`afternotok`, `--kill-on-invalid-dep=yes`), so it starts only after they ended without exit 0,
  and Slurm deletes it once one exits 0. A Task runs in at most one link at a time: never two
  executions of a Task.
- **Lease:** an Attempt with no Tasks of its own (`Kind.LEASE`), planned by `Campaign.hold` or
  `plan --hold N`: one Slurm job holding N **slots**, each a copy of its Profile's per-Task
  Resources, for its whole time limit. Its batch script serves orders until Slurm ends it, never
  exiting on its own, and after its pre-limit signal it takes no more. `max_links` queues a chain
  of leases. It needs `signal_before_end`.
- **Step:** an Attempt of one Task that runs inside a lease link (`Kind.STEP`, `within` the lease)
  as one exact `srun` step of the lease's job, using K of its slots. Planned by
  `Campaign.plan(lease=...)`, `plan --lease ID`, or `start`, always into the lease's current link:
  the first of its chain that Slurm lists, has not ended, and had no pre-limit signal. A step is
  link 1 of 1 of its own chain, writes its own start marker once Slurm gave it room, and stops being
  live as soon as it ended or was withdrawn.
- **Order:** `<log_root>/<lease>.d/<step>.run`, a step's `srun` line and run time, written by
  `submit` into the lease's owner-only order directory and sourced by the lease when it takes it.
- **Claim:** `<log_root>/<lease>.d/<step>.claim`, a symbolic link created without replacing that
  decides each order exactly once: `taken` when the lease won it, `withdrawn` when `cancel` or
  `reconcile` did. Only the winner acts.
- **Void marker:** `<step>.<job>.void`, written by the client after its claim on the step's order
  won: the step never runs and reads `STOPPED (never started)`. A step's stop request is
  `<step>.<job>.stop`, which the lease acts on with `SIGUSR1`, then `SIGTERM` after
  `signal_before_end`.
- **Handoff marker:** `<allocation_id>.<slot>.go`, written by a chain link into the next link's
  name once that slot's step returned and the slot continues (exit 75, or stopped by or a nonzero
  exit after the pre-limit signal). Its content is the Task's run time so far in seconds, for the
  next step. A later link runs only the slots handed to it.
- **Budget:** a Task's lifetime run-time limit across all its allocations, given by
  `plan --budget` (`Continuation`). A Task's own links are full `time_limit` links and one shorter
  last link that never add up to more; Tasks packed with different run times share links sized
  for the neediest, and the plan warns. A step learns it as `SERVATUS_BUDGET`, with `SERVATUS_RUN_TIME`,
  `SERVATUS_LINK`, and `SERVATUS_LINKS`; exit 75 (`YIELD_EXIT`) asks to continue.
- **Continuation (`Continuation`):** how far one plan continues its Tasks: an optional `budget`
  and an optional `max_links` (at most `MAX_LINKS`, 32). `Continuation()` is a plain plan.
- **Extension:** an explicit retry under a budget of a Task that already `SUCCEEDED`, or whose
  probe reports `VALID`; the plan names it (`Plan.extensions`). It is the only way to give a Task
  more time.
- **Run time (`RunTime`):** how long a Task has run, summed over every allocation that ran it,
  each the span between its run markers' mtimes. A step still running counts until now, and a
  `LOST` one counts its allocation's whole time limit; either makes the total incomplete. Gaps
  between allocations never count.
- **Capacity (`capacity`):** how many Tasks of one Resources fit one allocation under every Target
  ceiling, optionally lowered by `tasks_per_allocation`, and how many slots a lease may hold.
- **Slot:** a Task's zero-based position within its allocation.
- **Decision (`Decision`):** the reviewed part of a plan: Campaign identity and revision, Profile,
  selected, held, and deferred Tasks, packing, whether a probe was used, a random nonce from which
  allocation identities derive, the `Continuation`, each selected Task's measured run time
  (`spent`, with a continuation or for steps), and its extensions. `slots` alone makes it a lease
  plan; `slots` and `within` a step plan.
- **Plan (`Plan`):** a Decision (`plan.decision`) plus what it derives from the current state:
  `PlannedAllocation`s (Task keys, totals, batch script, `sbatch` argv), the digest, and `retry`
  (the selected or deferred Tasks that resubmit accepted work), with `warnings` for deferred work
  and for budgets a chain cannot spend as asked. A saved plan (`servatus.plan/4`) holds only the
  Decision and digest. Submission rebuilds the plan
  from its Decision and the current state and refuses it when it differs.
- **Hold (`Hold`):** why a Task is not selected: `VALID` (valid result), `UNRESOLVED` (intent
  without outcome), `SUBMITTED` (accepted work and not requested for retry; the scheduler is not
  consulted), `ACTIVE` (Slurm still lists earlier work of the Task, finished but not yet purged
  or a chain's pending links included, or a job of an allocation recorded as not submitted),
  `SPENT` (under a budget, the Task's run time leaves no room for one more link), or
  `NOT_REQUESTED` (outside `only`).
- **Deferred Task:** eligible work beyond the plan's `max_allocations_per_submit` batch, which
  counts every link of a chain; whole chains are deferred.
- **Retry selector (`Retry`):** a bulk retry choice. `Retry.FAILED` selects Tasks whose current
  execution is `FAILED` or `STOPPED`, never `LOST`; `Retry.INCOMPLETE` selects
  Tasks whose current execution is terminal and whose probed result is missing, and requires a
  probe. Explicit retry keys name single Tasks and are checked strictly (`PlanRefused` for unknown
  keys, valid results, Tasks never accepted, and live work); they accept `LOST`.
- **Attempt:** one durable allocation record: Profile, `Link`, Task keys, intent revision and time,
  then exactly one outcome revision with the accepted job, or with none for not submitted. Its
  `acceptance` follows from those, and an Attempt's retried Tasks are those accepted earlier. Its
  `slots` and `within` give its `Kind`: an allocation, a lease, or a step.
- **Link (`Link`):** an Attempt's place in its chain of allocations: the chain (the allocation id
  of link 1), its 1-based index, the chain's planned count, and the lifetime budget the chain was
  planned with. A plain allocation is link 1 of 1. Every later link follows the accepted previous
  link of its chain, with the same Tasks.
- **Intent:** the synced unresolved Attempt recorded before scheduler contact.
- **Receipt (`Receipt`):** a positive Slurm job number proving acceptance, not completion.
- **Unresolved allocation:** an intent without an outcome. It blocks its Tasks until `reconcile`
  resolves it: a queue row, else a run marker, records acceptance with its job; with neither, an
  intent at least 15 minutes old is recorded as not submitted, and a younger one is reported as
  waiting (`ReconcileResult`). An unresolved step resolves at once through its claim: accepted when
  the lease took its order, not submitted when the client's claim won.
- **Stray:** a job Slurm lists for an allocation recorded as not submitted. It holds its Tasks
  `ACTIVE` and `status` warns about it until it leaves the queue.
- **Submit result (`SubmitResult`):** receipts, unresolved, and unattempted allocations from one
  submission. `SubmissionInterrupted` carries it when submission stops early.
- **Shape check (`ShapeCheck`):** one time-specific `sbatch --test-only` answer for a distinct
  allocation shape.
- **Result probe:** a caller function that receives Tasks and returns the keys whose canonical
  results are valid. It is bound to a Campaign handle, called once per operation, and never stored.
- **Observation (`Observation`):** one accepted allocation's transient evidence: its `squeue`
  state while Slurm lists the job (`None` once purged; a terminal state is still settling and
  live), the mtimes of its start, stop, and limit markers, each slot's end (`StepEnd`: exit code
  and mtime), and which slots it was handed.
- **Execution (`Execution`):** a Task's state in its current allocation, from its slot:
  `QUEUED`, `RUNNING`, `ENDING` (Slurm reports the job ended but still lists it, and the slot
  has no end marker; its outcome is final once Slurm forgets the job), `SUCCEEDED` (exit 0),
  `FAILED` (another exit code), `STOPPED` (interrupted, never started, or exited 75), or `LOST`
  (started, then vanished without a stop marker). `ENDING` and `LOST` are not terminal. Exit codes
  accompany steps that returned. A `StopReason` says why a `STOPPED` Task stopped when the
  evidence says: `never started`, `yielded` (exit 75), `time limit`, or `budget spent`.
- **Observation scope:** the Attempts a plan needs to observe: the accepted and not-submitted
  Attempts of Tasks that could be retried, and the not-submitted Attempts of Tasks never accepted.
  Plans of fresh Tasks without such history observe nothing.
- **Status (`Status`):** one transient revision-bound projection of Tasks (`TaskStatus`) and
  Attempts (`AttemptStatus`), with counts, result readiness, and quiescence. A Task's
  `execution` is `UNRESOLVED` while one of its Attempts is, else the observed `Execution` of its
  current allocation (`ACCEPTED` when the scheduler was not observed), else `NOT_SUBMITTED`
  (never accepted). Each Task also shows its run time and its current Attempt's `Link`.
  `to_json()` exports it as a `servatus.status/4` document; `to_document` encodes other results
  for JSON.
- **Current allocation:** the latest accepted Attempt that was handed the Task: link 1 of a chain
  always is, and a later link once its handoff marker appeared. Status, retry selection, and
  `logs --task` all follow it.
- **Result readiness:** the sealed roster has valid results for every Task.
- **Quiescence:** scheduler evidence was requested, Slurm lists no job of any accepted allocation
  and no stray, and no acceptance is unresolved. It is independent from result readiness.
- **Cancellation:** `scancel --me --name=servatus-<allocation_id>` of the chosen accepted or
  not-submitted allocations that Slurm lists as queued or running, by allocation or for every such
  allocation of a Task, newest first. A chosen chain link brings every later link of its chain, and
  a lease ends with every step in it. A live step is stopped inside its lease instead: its order is
  withdrawn when the lease has not taken it, and otherwise the lease stops it. Every chosen
  allocation is tried. It is a request; it retries nothing.
- **Log snapshot (`LogSnapshot`):** a bounded binary suffix of an allocation or Task log. It is
  sensitive, untrusted, and has no lifecycle authority.

## Publication

- **Destination:** the application-owned canonical path. It is immutable once published.
  `DestinationExists` means a destination exists and nothing else; a child's destination lies
  inside its parent's private work.
- **Stage:** one private owner-only directory beside the destination where a result is assembled
  before its no-replace commit.
- **Draft (`Draft`):** the builder's handle on a directory stage. `link` and `link_tree` hard-link
  regular files into it; a link onto an occupied draft path is `ConfigurationError`. Using it after
  the builder returns raises `RuntimeError`.
- **File member:** the single entry named after the destination inside a `publish_file` stage. The
  writer may create it any way it likes; after the writer returns it must be a single-link regular
  file, and Servatus pins, chmods, syncs, and commits it.
- **Publication (`Publication`):** the committed destination plus whether private cleanup remains
  pending.
- **Mode:** the permission bits applied just before commit to every directory of a `publish` tree
  (file modes stay as written) or to a `publish_file` file (overriding the writer's choice). Stages
  are owner-only until then.
- **Workspace (`Workspace`):** stable, identity-bound owner-only private work retained when
  resumable work fails, removed after durable publication or by `discard()`.
- **Identity:** opaque application bytes whose digest and inode pins bind a Workspace to one logical
  request.
- **Child workspace:** one independently locked resumable result beneath a future parent
  destination.
- **Residue:** private work or stages left behind by interruption or unprovable cleanup. Entering a
  Workspace whose destination exists reclaims residue bound to the same identity (or never
  initialized), then raises `DestinationExists`. Containers are removed `work` first and identity
  last, so an interrupted removal leaves an identity-bound remnant that only the same identity
  finishes; work without an identity is never adopted.
- **Directory lock:** an exclusive `flock` coordinating entries of one directory (Workspace
  parents, the directory-commit fallback). Where a directory handle cannot be locked (NFS), an
  owner-only `.servatus.lock` file in the directory is locked instead and never removed.
- **Builder / writer:** the application callbacks that write and validate a draft or file member,
  then stop mutating it before returning.

Servatus owns lifecycle mechanics, not application meaning. Checkpoints, manifests, schemas,
validation rules, task selection, and scientific completion remain with the calling project.
