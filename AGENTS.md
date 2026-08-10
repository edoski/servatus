# Agent guidance

Servatus favors small, deep interfaces and ordinary typed Python. Keep application meaning outside
the package, reject unsafe filesystem behavior, and prefer clean breaks over compatibility layers.

Use synthetic temporary directories for tests. Never contact SSH, Slurm, Apptainer, external
storage, or an application's real outputs from the test suite.

Run pytest, Ruff, Pyright, Vulture, package builds, and an installed-artifact smoke before handing
off a change.
