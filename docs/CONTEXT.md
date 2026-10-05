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
- **Launcher:** how a step starts a Task. The **Apptainer launcher** (`Apptainer`) runs the Task in
  one immutable image with a clean environment; the **direct launcher** (no container) runs the
  Task's absolute `args[0]` under `env -i`.
- **Profile (`Profile`):** one nonbinding label plus a complete Target and Resources, loaded from
  `SERVATUS.toml` with document-level defaults overridden per key.
- **Transport:** how scheduler commands reach Slurm: `Ssh` (OpenSSH in batch mode, output fenced by
  per-call markers) when the Target has a host, `Local` otherwise. Both run with a scrubbed
  environment and fixed bounds; every failure surfaces as `Unavailable`.
- **Allocation:** one single-node Slurm job running its Tasks as concurrent exact steps. Each step
  is named `servatus-<allocation_id>-<slot>`.
- **Slot:** a Task's zero-based position within its allocation.
- **Plan (`Plan`):** one reviewed decision bound to a Campaign revision: Profile, selected, held,
  and deferred Tasks, retry choices, duplicate-risk acknowledgements, packing, a random nonce from
  which allocation identities derive, and a digest. Submission rebuilds the plan from its decision
  and the current state and refuses it when the digest differs.
- **Hold (`Hold`):** why a Task is not selected: `VALID`, `ACTIVE`, `UNRESOLVED`, `FINISHED`,
  `UNOBSERVABLE`, or `NOT_REQUESTED`.
- **Deferred Task:** eligible work beyond the plan's `max_allocations_per_submit` batch.
- **Retry selector (`Retry`):** a bulk retry choice. `Retry.FAILED` selects Tasks whose current
  work failed or was cancelled; `Retry.INCOMPLETE` selects every terminal accepted Task without a
  valid result and requires a probe. Explicit retry keys name single Tasks.
- **Duplicate-risk acknowledgement:** an explicit decision allowing retry when earlier accepted
  work is unobservable. It never claims the earlier work stopped.
- **Attempt:** one durable allocation record: Profile, Task keys, retry choices, intent revision
  and time, then exactly one outcome (accepted with a job, or not submitted).
- **Intent:** the synced unresolved Attempt recorded before scheduler contact.
- **Receipt (`Receipt`):** a Slurm job identity (`JobRef`) proving acceptance, not completion.
- **Unresolved allocation:** an intent without an outcome. It blocks its Tasks until `reconcile`,
  `mark_accepted`, or `mark_not_submitted` resolves it.
- **Submit result (`SubmitResult`):** receipts, unresolved, and unattempted allocations from one
  submission. `SubmissionInterrupted` carries it when submission stops early.
- **Shape check (`ShapeCheck`):** one time-specific `sbatch --test-only` answer for a distinct
  allocation shape.
- **Result probe:** a caller function that receives Tasks and returns the keys whose canonical
  results are valid. It is bound to a Campaign handle, called once per operation, and never stored.
- **Scheduler evidence (`SchedulerEvidence`):** transient allocation state normalized from `squeue`
  and anchored `sacct` history to queued, running, succeeded, failed, cancelled, or unknown, with
  raw state, exit code, reason, whether Slurm retains the work, and any per-allocation `problem`.
- **Step evidence (`StepEvidence`):** the state and exit code of one Task's own `srun` step, when
  accounting reports it. It refines a packed Task's execution state.
- **Observation scope:** the accepted Attempts a plan needs to observe: only those whose Tasks
  could be retried. Plans of fresh Tasks observe nothing.
- **Status (`Status`):** one transient revision-bound projection of Tasks (`TaskStatus`) and
  Attempts (`AttemptStatus`), with result readiness and quiescence. `to_json()` exports it.
- **Result readiness:** the sealed roster has valid results for every Task.
- **Quiescence:** scheduler evidence was requested, every accepted Attempt is terminal, and no
  acceptance is unresolved. It is independent from result readiness.
- **Log snapshot (`LogSnapshot`):** a bounded binary suffix of an allocation or Task log. It is
  sensitive, untrusted, and has no lifecycle authority.

## Publication

- **Destination:** the application-owned canonical path. It is immutable once published.
- **Stage:** one private owner-only directory beside the destination where a result is assembled
  before its no-replace commit.
- **Draft (`Draft`):** the builder's handle on a directory stage. `link` and `link_tree` hard-link
  regular files into it. It is invalid once the builder returns.
- **File member:** the single entry named after the destination inside a `publish_file` stage. The
  writer may create it any way it likes; Servatus pins, syncs, and commits it after the writer
  returns.
- **Publication (`Publication`):** the committed destination plus whether private cleanup remains
  pending.
- **Mode:** the permission bits applied to a published entry just before commit; stages are
  owner-only until then.
- **Workspace (`Workspace`):** stable, identity-bound owner-only private work retained when
  resumable work fails, removed after durable publication or by `discard()`.
- **Identity:** opaque application bytes whose digest and inode pins bind a Workspace to one logical
  request.
- **Child workspace:** one independently locked resumable result beneath a future parent
  destination.
- **Residue:** private work or stages left behind by interruption or unprovable cleanup. Entering a
  Workspace whose destination exists reclaims its residue.
- **Retained tree:** one existing owner-only destination sibling removed only after the destination
  commit is durable.
- **Builder / writer:** the application callbacks that write and validate a draft or file member,
  then stop mutating it before returning.

Servatus owns lifecycle mechanics, not application meaning. Checkpoints, manifests, schemas,
validation rules, task selection, and scientific completion remain with the calling project.
