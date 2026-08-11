# Servatus context

Servatus uses a small generic vocabulary:

- **Task:** one stable opaque key, argument vector, and byte payload.
- **Campaign:** one append-only ordered task sequence and its durable submission history; `open`
  registers the complete authored roster while `load` reopens existing state without authoring it.
- **Resource request:** one homogeneous per-Task CPU, MiB, whole-GPU, and wall-time requirement.
- **Target:** one concrete SSH/Slurm/Apptainer route with conservative request ceilings.
- **Plan:** a local immutable selection whose exported document is accepted only when regeneration
  from its typed inputs produces the same canonical bytes.
- **Allocation:** one single-node Slurm job containing concurrent exact Task steps.
- **Intent:** the synced record written before possible scheduler acceptance; it retains ordered
  Task keys, plan and script digests, exact command, allocation totals, and query window.
- **Receipt:** a positive Slurm job identity proving scheduler acceptance, not completion; its Task
  keys come from the immutable intent.
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
- **Draft:** one unique, disposable directory assembled before publication.
- **File stage:** one unique, empty regular file written and validated in place before publication.
- **Publication:** the committed destination plus whether private cleanup remains pending.
- **Builder:** the application callback that writes and validates a draft before returning.
- **Writer:** the application callback that writes and validates a file stage before returning.

Servatus owns lifecycle mechanics, not application meaning. Checkpoints, manifests, schemas,
validation rules, task selection, and scientific completion remain with the calling project.
