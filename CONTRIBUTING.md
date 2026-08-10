# Contributing

Servatus keeps its public surface deliberately small. Changes should solve a demonstrated lifecycle
problem without importing application schemas, workflow topology, or backend plugin machinery.

Set up and verify the repository with:

```sh
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run vulture
uv build
```

Tests must use synthetic temporary directories. Pull requests must not depend on a live scheduler,
cluster, container runtime, or external data store.
