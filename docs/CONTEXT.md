# Servatus context

Servatus uses a small generic vocabulary:

- **Task:** one stable opaque key, argument vector, and byte payload.
- **Campaign:** one append-only ordered task sequence and its durable submission history; every
  registered prefix value remains immutable.
- **Resource request:** one homogeneous per-Task CPU, MiB, whole-GPU, and wall-time requirement.
- **Target:** one concrete SSH/Slurm/Apptainer route with conservative request ceilings.
- **Plan:** a local immutable selection, grouping, script, command, and digest snapshot.
- **Allocation:** one single-node Slurm job containing concurrent exact Task steps.
- **Intent:** the synced record written before possible scheduler acceptance.
- **Receipt:** a positive Slurm job identity proving scheduler acceptance, not completion.
- **Ambiguous allocation:** an intent without a receipt or explicit operator resolution.
- **Destination:** the application-owned canonical path. It is immutable once published.
- **Workspace:** stable, identity-bound owner-only private state retained when resumable work fails;
  Servatus durably initializes and exactly cleans its hidden lifecycle tree.
- **Child workspace:** one independently locked resumable result beneath a future parent destination.
- **Identity:** opaque application bytes whose digest binds a workspace to one logical request.
- **Draft:** one unique, disposable directory assembled before publication.
- **File stage:** one unique, empty regular file written and validated in place before publication.
- **Publication:** the committed destination plus whether private cleanup remains pending.
- **Builder:** the application callback that writes and validates a draft before returning.
- **Writer:** the application callback that writes and validates a file stage before returning.

Servatus owns lifecycle mechanics, not application meaning. Checkpoints, manifests, schemas,
validation rules, task selection, and scientific completion remain with the calling project.
