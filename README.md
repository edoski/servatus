# Servatus

Run resumable work through Slurm and atomically publish validated outputs.

Servatus 0.0.1 is a publication-only alpha. It provides a small Python interface for keeping
private resumable work and exposing an application-built directory atomically. Slurm campaigns
arrive in a later release.

```sh
pip install servatus
```

## Publication

Use `publish` when failed work is disposable:

```python
from pathlib import Path

from servatus import Draft, publish


def build(draft: Draft) -> None:
    (draft.path / "result.json").write_text('{"status":"complete"}\n')


publication = publish(Path("outputs/run-1"), build)
```

Use `Workspace` when a worker must retain private checkpoints across restarts:

```python
from servatus import Draft, Workspace

destination = Path("outputs/model-1")
with Workspace(destination, identity=b"model request bytes") as workspace:
    checkpoint = workspace.path / "last.ckpt"
    # The application creates or resumes its own checkpoint here.

    def assemble(draft: Draft) -> None:
        draft.link(checkpoint, "last.ckpt")
        # Perform application validation before returning.

    publication = workspace.publish(assemble)
```

`Workspace` binds its stable hidden state to the SHA-256 digest of the opaque identity and holds a
nonblocking writer lock. `Draft.link` only hard-links regular files into a safe relative path. The
application must not mutate a linked source inode after `Draft.link()` returns and before
publication completes. It owns file contents, validation, schemas, and completion meaning.

## Guarantees

- A destination is absent or one complete directory; it is never overwritten.
- Work, hard-link sources, stages, and destination must share a filesystem.
- Files and directories are synced before a kernel-exclusive commit; the destination parent is
  synced afterward.
- Builder failures expose no destination. Resumable work remains available, while disposable
  stages are removed.
- Successful workspace publication removes private state. A cleanup failure returns
  `Publication(cleanup_pending=True)` without misreporting the committed destination as failed.
- Symlinks, special files, escaping link paths, path substitution, and unsupported exclusive-rename
  primitives fail closed.

Servatus supports POSIX filesystems on Linux and macOS. Hardware durability still depends on the
filesystem and mount. It is not a security boundary against another process that can arbitrarily
modify the same parent directory.

## Non-goals in 0.0.1

Servatus does not interpret checkpoints or ML artifacts, decide when work is valid, model workflow
graphs, migrate old outputs, copy across filesystems, overwrite destinations, or provide scheduler
execution. It has no plugins, callbacks beyond the one build seam, runtime dependencies, daemon,
database, or global run store.

The public API is `Draft`, `Workspace`, `Publication`, `publish`, and the compact errors exported by
`servatus`. See [the context glossary](docs/CONTEXT.md) and [architecture decisions](docs/adr/README.md)
for the ownership boundary.
