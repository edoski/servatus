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
- **Resources (`Resources`):** one homogeneous per-Task CPU, MiB, whole-GPU, wall-time, and optional
  `signal_before_end` request for one plan.
- **Target (`Target`):** one concrete Slurm route: optional SSH host, absolute Slurm directory,
  work and log roots, partitions, site options, optional launcher, and conservative ceilings. Every
  Attempt keeps its original Target for observation, recovery, logs, and cancellation.
- **Launcher:** how a step starts a Task. The **Apptainer launcher** (`Apptainer`) runs the Task's
  `args` with `apptainer exec` in one immutable image with a clean environment (the image's
  runscript is never used; `args` must be nonempty); the **direct launcher** (no container) runs
  the Task's absolute `args[0]` under `env -i`.
- **Profile (`Profile`):** one nonbinding label plus a complete Target and Resources, loaded from
  `SERVATUS.toml` with document-level defaults overridden per key.
- **Transport (`Transport`):** how scheduler commands reach Slurm. `connect(target)` returns an SSH
  transport (OpenSSH in batch mode, output fenced by per-call markers) when the Target has a host,
  and a local one otherwise. Both run with a scrubbed environment and fixed bounds and return a
  `Completed`; every failure surfaces as `Unavailable`. Any callable from Target to Transport (for
  example `servatus.testing.FakeScheduler`) can replace `connect`. `ping(target)` proves a Target's
  scheduler answers (`sbatch --version`).
- **Allocation:** one single-node Slurm job named `servatus-<allocation_id>` running its Tasks as
  concurrent exact steps. Each step is named `servatus-<allocation_id>-<slot>`.
- **Capacity (`capacity`):** how many Tasks of one Resources fit one allocation under every Target
  ceiling, optionally lowered by `tasks_per_allocation`.
- **Slot:** a Task's zero-based position within its allocation.
- **Decision (`Decision`):** the reviewed part of a plan: Campaign identity and revision, Profile,
  selected, held, and deferred Tasks, retry choices, duplicate-risk acknowledgements, packing,
  whether a probe was used, and a random nonce from which allocation identities derive.
- **Plan (`Plan`):** a Decision (`plan.decision`) plus its derived `PlannedAllocation`s (Task
  keys, totals, batch script, `sbatch` argv) and digest, with `warnings` for acknowledged duplicate
  risk and deferred work. A saved plan (`servatus.plan/1`) holds only the Decision and digest.
  Submission rebuilds the plan from its Decision and the current state and refuses it when it
  differs.
- **Hold (`Hold`):** why a Task is not selected: `VALID` (valid result), `UNRESOLVED` (intent
  without outcome), `SUBMITTED` (accepted work and not requested for retry; the scheduler is not
  consulted), `ACTIVE` (retry requested, but Slurm still holds earlier work), `UNOBSERVABLE`
  (evidence for earlier work is missing, or the Task's execution is `UNKNOWN`), or `NOT_REQUESTED`
  (outside `only`).
- **Deferred Task:** eligible work beyond the plan's `max_allocations_per_submit` batch.
- **Retry selector (`Retry`):** a bulk retry choice. `Retry.FAILED` selects Tasks whose current
  (latest accepted) execution failed or was cancelled; `Retry.INCOMPLETE` selects terminal Tasks
  whose probed result is missing and requires a probe. Explicit retry keys name single Tasks and are
  checked strictly (`PlanRefused`).
- **Duplicate-risk acknowledgement:** an explicit decision, for an explicitly retried key only,
  allowing retry when earlier accepted work has `UNKNOWN` evidence. It never claims the earlier
  work stopped.
- **Attempt:** one durable allocation record: Profile, Task keys, retry choices, intent revision
  and time, then exactly one outcome (accepted with a job, or not submitted).
- **Intent:** the synced unresolved Attempt recorded before scheduler contact.
- **Receipt (`Receipt`):** a Slurm job identity (`JobRef`) proving acceptance, not completion.
- **Unresolved allocation:** an intent without an outcome. It blocks its Tasks until `reconcile`,
  `mark_accepted`, or `mark_not_submitted` resolves it. `mark_not_submitted` records only proven
  absence: the original target must report no job carrying the allocation's identity.
- **Submit result (`SubmitResult`):** receipts, unresolved, and unattempted allocations from one
  submission. `SubmissionInterrupted` carries it when submission stops early.
- **Shape check (`ShapeCheck`):** one time-specific `sbatch --test-only` answer for a distinct
  allocation shape.
- **Result probe:** a caller function that receives Tasks and returns the keys whose canonical
  results are valid. It is bound to a Campaign handle, called once per operation, and never stored.
- **Scheduler evidence (`SchedulerEvidence`):** transient allocation state normalized from `squeue`
  and anchored `sacct` history to an `AllocationState` (queued, running, succeeded, failed,
  cancelled, or unknown), with raw state, exit code, reason, whether Slurm retains the work, and
  any per-allocation `problem` (for example, a foreign job now holding the job number).
- **Step evidence (`StepEvidence`):** the state and exit code of one Task's own `srun` step, when
  accounting reports it. An observation holds one entry per slot (`None`: step not found), or none
  at all when the step query failed (step evidence unavailable).
- **Task execution:** a Task's state in its current allocation: its own step when known, else the
  allocation's state. `UNKNOWN` or contradictory (`problem`) allocation evidence makes it `UNKNOWN`
  whatever the steps say, and so does a failed or cancelled allocation of several Tasks without the
  Task's step (that Task may have finished first). Exit codes accompany terminal executions only.
- **Observation scope:** the accepted Attempts a plan needs to observe: only those whose Tasks
  could be retried. Plans of fresh Tasks observe nothing.
- **Status (`Status`):** one transient revision-bound projection of Tasks (`TaskStatus`) and
  Attempts (`AttemptStatus`), with counts, result readiness, and quiescence. `to_json()` exports
  it as a `servatus.status/1` document; `to_document` encodes other results for JSON.
- **Result readiness:** the sealed roster has valid results for every Task.
- **Quiescence:** scheduler evidence was requested, every accepted Attempt is terminal and no
  longer retained by Slurm, and no acceptance is unresolved. It is independent from result
  readiness.
- **Cancellation:** `scancel` of accepted allocations not known to be terminal, by allocation or
  for every accepted allocation of a Task. Every chosen allocation is tried. It is a request; it
  retries nothing.
- **Log snapshot (`LogSnapshot`):** a bounded binary suffix of an allocation or Task log. It is
  sensitive, untrusted, and has no lifecycle authority.

## Publication

- **Destination:** the application-owned canonical path. It is immutable once published.
- **Stage:** one private owner-only directory beside the destination where a result is assembled
  before its no-replace commit.
- **Draft (`Draft`):** the builder's handle on a directory stage. `link` and `link_tree` hard-link
  regular files into it. Using it after the builder returns raises `RuntimeError`.
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
- **Retained tree:** one existing owner-only destination sibling removed only after the destination
  commit is durable.
- **Builder / writer:** the application callbacks that write and validate a draft or file member,
  then stop mutating it before returning.

Servatus owns lifecycle mechanics, not application meaning. Checkpoints, manifests, schemas,
validation rules, task selection, and scientific completion remain with the calling project.
