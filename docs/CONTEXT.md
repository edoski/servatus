# Servatus context

Servatus uses a small generic vocabulary:

- **Task:** one stable opaque key, argument vector, and byte payload.
- **Campaign:** one ordered Task roster and its durable Attempt history. `create` authors a fixed
  roster, or an appendable one when requested; `load` reopens it; `append` registers only new Tasks;
  and `seal` ends authoring irreversibly.
- **Resource request:** one homogeneous per-Task CPU, MiB, whole-GPU, and wall-time requirement for
  one plan. Each Attempt retains the request used for its submission.
- **Target:** one concrete SSH/Slurm/Apptainer route with conservative request ceilings. Every
  Attempt retains its original route for observation, recovery, and diagnostics.
- **Profile:** one nonbinding label plus a complete Target and Resource request.
- **Plan:** one compact reviewed decision bound to a Campaign revision: execution configuration,
  selected allocations, retry choices, result-probe requirement, and an integrity digest.
- **Excluded task:** work withheld by the eligibility policy.
- **Deferred task:** eligible work beyond the plan's bounded allocation batch.
- **Allocation:** one single-node Slurm job containing concurrent exact Task steps.
- **Attempt:** one durable allocation record with its exact execution configuration, Task keys,
  retry choices, intent timestamp and revision, and unresolved, accepted, or
  not-submitted outcome. Reconciliation windows derive from the timestamp; outcome revisions
  preserve actual mutation chronology.
- **Intent:** the synced unresolved Attempt recorded before scheduler contact.
- **Receipt:** a positive Slurm job identity proving acceptance, not completion; its Task keys come
  from the durable intent.
- **Submit result:** structured receipts and unresolved or unattempted allocations from one reviewed
  batch. A receipt observed but not durably recorded is distinguished from confirmed receipts.
- **Result probe:** an ephemeral synchronous caller function that validates a canonical result for
  one opaque Task without exposing the application schema to Servatus.
- **Allocation evidence:** transient scheduler evidence normalized to queued, running, succeeded,
  failed, cancelled, or unknown. Exact retained-work evidence blocks retry; anchored accounting
  establishes terminal evidence and identifies later requeue incarnations.
- **Campaign view:** one transient revision-bound projection of all Attempts, current Task
  execution and result evidence, readiness, and quiescence. Planning collects its own observations.
- **Operational record:** a redacted JSON diagnostic projection of Campaign identity, roster,
  Attempt chronology, receipts, and normalized scheduler evidence. It is identifying,
  nonauthoritative, and never automatically published.
- **Log snapshot:** a transient bounded binary suffix from an accepted Attempt's allocation or Task
  log. It has no lifecycle authority.
- **Result readiness:** the sealed roster has valid immutable caller results for every Task.
- **Quiescence:** scheduler evidence was requested, every accepted Attempt is proven terminal, and
  no acceptance remains unresolved. It is independent from result readiness.
- **Duplicate-risk acknowledgement:** an explicit operator decision allowing retry when accepted
  scheduler evidence is unknown. It never claims that earlier work stopped.
- **Validation result:** time-specific bounded Slurm `--test-only` validation of a current plan.
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
