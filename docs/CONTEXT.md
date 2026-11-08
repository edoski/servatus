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
  writable roots, and a one-Task `sbatch --test-only`.
- **Allocation:** one single-node Slurm job named `servatus-<allocation_id>`, submitted with
  `--no-requeue`, running its Tasks as concurrent exact steps. Each step is named
  `servatus-<allocation_id>-<slot>`.
- **Run marker:** an empty file the batch script creates in `log_root`:
  `<allocation_id>.<job>.start` before any step, `<allocation_id>.<job>.<slot>.<exit code>.end`
  when that slot's step returns, and `<allocation_id>.<job>.stop` when the script is interrupted.
- **Capacity (`capacity`):** how many Tasks of one Resources fit one allocation under every Target
  ceiling, optionally lowered by `tasks_per_allocation`.
- **Slot:** a Task's zero-based position within its allocation.
- **Decision (`Decision`):** the reviewed part of a plan: Campaign identity and revision, Profile,
  selected, held, and deferred Tasks, packing, whether a probe was used, and a random nonce from
  which allocation identities derive.
- **Plan (`Plan`):** a Decision (`plan.decision`) plus what it derives from the current state:
  `PlannedAllocation`s (Task keys, totals, batch script, `sbatch` argv), the digest, and `retry`
  (the selected or deferred Tasks that resubmit accepted work), with `warnings` for deferred work.
  A saved plan (`servatus.plan/2`) holds only the Decision and digest. Submission rebuilds the plan
  from its Decision and the current state and refuses it when it differs.
- **Hold (`Hold`):** why a Task is not selected: `VALID` (valid result), `UNRESOLVED` (intent
  without outcome), `SUBMITTED` (accepted work and not requested for retry; the scheduler is not
  consulted), `ACTIVE` (Slurm still lists earlier work of the Task, finished but not yet purged
  included, or a job of an allocation recorded as not submitted), or `NOT_REQUESTED` (outside
  `only`).
- **Deferred Task:** eligible work beyond the plan's `max_allocations_per_submit` batch.
- **Retry selector (`Retry`):** a bulk retry choice. `Retry.FAILED` selects Tasks whose current
  (latest accepted) execution is `FAILED` or `STOPPED`, never `LOST`; `Retry.INCOMPLETE` selects
  Tasks whose current execution is terminal and whose probed result is missing, and requires a
  probe. Explicit retry keys name single Tasks and are checked strictly (`PlanRefused` for unknown
  keys, valid results, Tasks never accepted, and live work); they accept `LOST`.
- **Attempt:** one durable allocation record: Profile, `Link`, Task keys, intent revision and time,
  then exactly one outcome revision with the accepted job, or with none for not submitted. Its
  `acceptance` follows from those, and an Attempt's retried Tasks are those accepted earlier.
- **Link (`Link`):** an Attempt's place in its chain of allocations: the chain (the allocation id
  of link 1), its 1-based index, the chain's planned count, and the lifetime budget the chain was
  planned with. A plain allocation is link 1 of 1. Every later link follows the accepted previous
  link of its chain, with the same Tasks.
- **Intent:** the synced unresolved Attempt recorded before scheduler contact.
- **Receipt (`Receipt`):** a positive Slurm job number proving acceptance, not completion.
- **Unresolved allocation:** an intent without an outcome. It blocks its Tasks until `reconcile`
  resolves it: a queue row, else a run marker, records acceptance with its job; with neither, an
  intent at least 15 minutes old is recorded as not submitted, and a younger one is reported as
  waiting (`ReconcileResult`).
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
  live), whether its start and stop markers exist, and one exit code per slot from its end markers.
- **Execution (`Execution`):** a Task's state in its current allocation, from its slot:
  `QUEUED`, `RUNNING`, `ENDING` (Slurm reports the job ended but still lists it, and the slot
  has no end marker; its outcome is final once Slurm forgets the job), `SUCCEEDED` or `FAILED`
  (from the end marker's exit code), `STOPPED` (interrupted, or never started), or `LOST`
  (started, then vanished without a stop marker). `ENDING` and `LOST` are not terminal.
  Exit codes accompany `SUCCEEDED` and `FAILED` only.
- **Observation scope:** the Attempts a plan needs to observe: the accepted and not-submitted
  Attempts of Tasks that could be retried, and the not-submitted Attempts of Tasks never accepted.
  Plans of fresh Tasks without such history observe nothing.
- **Status (`Status`):** one transient revision-bound projection of Tasks (`TaskStatus`) and
  Attempts (`AttemptStatus`), with counts, result readiness, and quiescence. A Task's
  `execution` is `UNRESOLVED` while one of its Attempts is, else the observed `Execution` of its
  latest accepted Attempt (`ACCEPTED` when the scheduler was not observed), else `NOT_SUBMITTED`
  (never accepted). `to_json()` exports it as a `servatus.status/2` document; `to_document` encodes
  other results for JSON.
- **Result readiness:** the sealed roster has valid results for every Task.
- **Quiescence:** scheduler evidence was requested, Slurm lists no job of any accepted allocation
  and no stray, and no acceptance is unresolved. It is independent from result readiness.
- **Cancellation:** `scancel --me --name=servatus-<allocation_id>` of the chosen accepted or
  not-submitted allocations that Slurm lists as queued or running, by allocation or for every such
  allocation of a Task. Every chosen allocation is tried. It is a request; it retries nothing.
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
