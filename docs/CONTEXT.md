# Servatus context

Servatus uses a small generic vocabulary:

- **Destination:** the application-owned canonical path. It is immutable once published.
- **Workspace:** stable, identity-bound private state retained when resumable work fails.
- **Identity:** opaque application bytes whose digest binds a workspace to one logical request.
- **Draft:** one unique, disposable directory assembled before publication.
- **Publication:** the committed destination plus whether private cleanup remains pending.
- **Builder:** the application callback that writes and validates a draft before returning.

Servatus owns lifecycle mechanics, not application meaning. Checkpoints, manifests, schemas,
validation rules, task topology, and scientific completion remain with the calling project.
