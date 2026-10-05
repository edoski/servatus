# Contributing

Servatus keeps its public surface deliberately small. Changes should solve a demonstrated lifecycle
problem without importing application schemas, workflow topology, or backend plugin machinery.
Read [AGENTS.md](AGENTS.md), [docs/CONTEXT.md](docs/CONTEXT.md), and the
[architecture decisions](docs/adr/README.md) before changing behaviour.

## Setup and verification

```sh
uv sync --locked
```

Run the full verification before handing off a change; CI runs the same steps on Python 3.11 to
3.14 on Linux and on 3.11 and 3.14 on macOS:

```sh
uv run coverage run -m pytest && uv run coverage report --show-missing
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run vulture
uv build
uv run --isolated --no-project --with dist/servatus-*.whl \
  python .github/scripts/smoke.py dist/servatus-*.whl
```

The last step installs the built wheel into a throwaway environment and checks the wheel contents
(`py.typed`, no tests), the public names, that `servatus` is imported from the installed wheel,
the console script and `--version`, an offline CLI campaign, a `FakeScheduler` plan, validate,
submit, and retry round-trip, and publication. It takes the wheel path as its only argument. Clear
`dist/` first if it holds older wheels. Coverage settings (`source`, branch coverage) live in
`pyproject.toml`.

## Test layout

- `tests/support/`: shared builders for synthetic Tasks, Targets, Resources, and Profiles.
- `tests/core/`: codec, configuration, state invariant and transitions, store, status, policy.
- `tests/scheduler/`: transport bounds, batch scripts, evidence parsing, and `FakeScheduler`.
- `tests/publication/`: publish, `publish_file`, Draft, Workspace, children, and POSIX fallbacks.
- `tests/faults/`: injected failures, interruptions, and crash points.
- `tests/campaign/`: Campaign behaviour and the CLI, driven through `FakeScheduler`.

`tests/` is on the import path, so tests import helpers as `from support.builders import target`.

## Rules

- Use synthetic temporary directories only. Never contact SSH, Slurm, Apptainer, external storage,
  or an application's real outputs.
- Drive Campaign tests through `servatus.testing.FakeScheduler` passed as `connect`; do not
  monkeypatch remote calls.
- Every `pytest.raises` uses `match=` so a test fails for the right reason.
- Test file basenames are unique across the whole `tests/` tree (pytest's prepend import mode).
- Pyright runs in strict mode over `src` and `tests`; a test file may opt down with a
  `# pyright: standard` first line.
- No runtime dependencies. Keep Python 3.11 compatibility (no PEP 695 syntax).
- Prefer clean breaks over compatibility layers, and record user-visible changes in
  [CHANGELOG.md](CHANGELOG.md).
