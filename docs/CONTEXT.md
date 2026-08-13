# Servatus context

Servatus uses a small generic vocabulary:

- **Task:** one stable opaque key, argument vector, and byte payload.
- **Campaign:** one ordered Task roster and its durable submission history; `open` authors an exact
  roster or ordered suffix while the roster is open, `seal` ends authoring irreversibly, and `load`
  reopens existing state without authoring it.
- **Roster phase:** `OPEN` permits exact ordered suffixes and execution; `SEALED` permits execution
  but no roster change.
- **Resource request:** one homogeneous per-Task CPU, MiB, whole-GPU, and wall-time requirement.
- **Target:** one concrete SSH/Slurm/Apptainer route with conservative request ceilings.
- **Plan:** a local immutable selection whose exported document is accepted only when regeneration
  from its typed inputs produces the same canonical bytes.
- **Allocation:** one single-node Slurm job containing concurrent exact Task steps.
- **Attempt:** one durable allocation record written before possible scheduler acceptance; it owns
  ordered Task and retry keys, Campaign revision, lineage, plan and script digests, exact command,
  allocation totals, query window, and one unresolved, accepted, or not-submitted outcome.
- **Intent:** the durably synced unresolved Attempt written before possible scheduler acceptance.
- **Receipt:** a positive Slurm job identity proving scheduler acceptance, not completion; its Task
  keys come from the immutable intent.
- **Result probe:** one ephemeral synchronous caller function that validates the canonical result
  for one opaque Task as valid, missing, or invalid without exposing its schema to Servatus.
- **Allocation evidence:** one transient time-stamped observation of an exact accepted receipt,
  normalized to queued, running, succeeded, failed, cancelled, or unknown.
- **Campaign view:** one immutable revision-bound projection containing every Attempt, current Task
  execution and result evidence, readiness, and quiescence without persisting observations.
- **Log snapshot:** one transient time-stamped bounded binary suffix from the allocation or packed
  Task log of one exact accepted Attempt; it is diagnostic evidence with no lifecycle authority.
- **Result readiness:** the sealed roster has valid immutable caller results for every Task.
- **Quiescence:** scheduler evidence was requested, every accepted Attempt is proven terminal, and
  no acceptance remains unresolved; it is independent from result readiness.
- **Unaccepted task:** one Task without a proven scheduler-acceptance receipt. Application
  completion is a separate caller-owned decision.
- **Validation result:** one immutable, time-specific result from bounded Slurm `--test-only`
  validation of an authoritative Campaign plan.
- **Ambiguous allocation:** an intent without a receipt or explicit operator resolution.
- **Destination:** the application-owned canonical path. It is immutable once published.
- **Workspace:** stable, identity-bound owner-only private state retained when resumable work fails;
  Servatus durably initializes and exactly cleans its hidden lifecycle tree.
- **Child workspace:** one independently locked resumable result beneath a future parent destination.
- **Identity:** opaque application bytes whose digest and three inode pins bind a workspace to one
  logical request and its container, lock, and work entries.
- **Draft:** one unique, disposable, owner-only directory assembled before publication; its trusted
  namespace remains quiescent while a hard link selects and validates one regular source inode.
- **File stage:** one unique, empty regular file written and validated in place before publication.
- **Publication:** the committed destination plus whether private cleanup remains pending.
- **Retained tree:** one existing owner-only destination sibling pinned before a directory build and
  removed only after the destination commit is durable.
- **Builder:** the application callback that writes and validates a draft, then stops mutating it
  before returning.
- **Writer:** the application callback that writes and validates a file stage, then stops mutating it
  before returning.

Servatus owns lifecycle mechanics, not application meaning. Checkpoints, manifests, schemas,
validation rules, task selection, and scientific completion remain with the calling project.
Builders, writers, and retained trees must be quiescent at their documented handoff points;
Servatus provides durable publication and safe commit-first retirement, not a writer lease or
application finalizer.
