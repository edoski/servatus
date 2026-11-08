"""Slurm operations over one ``Transport``: submit, test, read the queue and run markers, tail,
cancel, and write or withdraw a lease's orders.

Every method issues fixed, absolute argument vectors and parses replies with the pure functions in
``_evidence``. Command failures raise ``Unavailable``; untrustworthy replies raise
``EvidenceConflict``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath

from ..errors import ConfigurationError, EvidenceConflict, Unavailable
from ._config import Target
from ._evidence import Marker, QueueRow, parse_markers, parse_queue, parse_receipt
from ._remote import MAX_STREAM_BYTES, Completed, Transport, diagnostic
from ._script import MAX_ARGS, job_name, order_directory, owned_directory

MAX_QUERY_NAMES = 100
TAIL = "/usr/bin/tail"
FIND = "/usr/bin/find"
WRITABLE_DIRECTORY = ("/bin/sh", "-c", 'test -d "$1" && test -w "$1"', "servatus")
SQUEUE_OPTIONS = ("--me", "--all", "--states=all", "--noheader")
SQUEUE_FORMAT = "--format=%i|%j|%T"
MARKER_SUFFIXES = (".start", ".stop", ".limit", ".void", ".end", ".go")
MARKER_FORMAT = "%T@ %p\n"
_ORDERS = f"umask 077; d=$1; {owned_directory('d')} || exit 1; "
ORDER = (
    "/bin/sh",
    "-c",
    _ORDERS + 'cat > "$d/.$2.part" && mv -f -- "$d/.$2.part" "$d/$2.run"',
    "servatus-order",
)
# The claim is a symbolic link created without replacing; the lease takes an order under the
# same name, so exactly one side wins, and `readlink` tells either side which.
WITHDRAW = (
    "/bin/sh",
    "-c",
    _ORDERS + '[ -z "$4" ] || [ -e "$4" ] || command : > "$4" || exit 1; '
    'ln -s withdrawn "$d/$2.claim" 2>/dev/null; '
    'case $(readlink "$d/$2.claim" 2>/dev/null) in '
    'withdrawn) command : > "$3" && echo withdrawn ;; taken) echo taken ;; *) exit 1 ;; esac',
    "servatus-withdraw",
)
_MARKER_LINE_BYTES = len("0000000000.0000000000 /") + 24 + len(".4294967295.4294967295.255.end")


class CommandFailed(Unavailable):
    """The command ran and reported failure, unlike a transport that never got an answer."""


def _require_ok(completed: Completed, what: str, *, quiet: bool = True) -> Completed:
    """``completed`` if it exited 0 (and, when ``quiet``, wrote nothing to stderr)."""
    if completed.returncode != 0 or (quiet and completed.stderr):
        raise CommandFailed(f"{what} failed with {diagnostic(completed)}")
    return completed


def find_argv(log_root: PurePosixPath, patterns: Sequence[str]) -> tuple[str, ...]:
    """List the regular files in ``log_root`` whose names match any of ``patterns``, each with
    its mtime. ``-H`` follows ``log_root`` itself when it is a symbolic link, as Slurm and the
    batch script do."""
    names: list[str] = []
    for pattern in patterns:
        names.extend(("-o", "-name", pattern))
    return (
        FIND,
        "-H",
        str(log_root),
        "-maxdepth",
        "1",
        "-type",
        "f",
        "(",
        *names[1:],
        ")",
        "-printf",
        MARKER_FORMAT,
    )


MAX_NAME_PATTERNS = (MAX_ARGS - len(find_argv(PurePosixPath("/"), ("*",)))) // 3 + 1
"""How many name patterns one listing holds within ``check_command``'s argument bound."""


def marker_patterns(
    log_root: PurePosixPath, task_counts: Mapping[str, int]
) -> list[tuple[str, ...]]:
    """The name patterns of each directory read for these allocations' markers.

    A few allocations are named exactly (``<id>.*``; logs are ``<id>-...``), so the read stays
    small however many other markers share ``log_root``. Many allocations take one read of every
    marker, split by leading hex digit when their own markers could exceed the stream bound.
    """
    allocation_ids = sorted(task_counts)
    if len(allocation_ids) <= MAX_NAME_PATTERNS:
        return [tuple(f"{allocation_id}.*" for allocation_id in allocation_ids)]
    estimate = sum(
        (3 + 2 * count) * (len(str(log_root)) + _MARKER_LINE_BYTES)
        for count in task_counts.values()
    )
    prefixes = [""]
    if estimate > MAX_STREAM_BYTES // 2:
        prefixes = sorted({allocation_id[0] for allocation_id in allocation_ids})
    return [tuple(f"{prefix}*{suffix}" for suffix in MARKER_SUFFIXES) for prefix in prefixes]


class Scheduler:
    """Slurm commands of one ``target``, run through ``transport``."""

    def __init__(self, transport: Transport, target: Target) -> None:
        self._transport = transport
        self._target = target

    def _command(self, name: str) -> str:
        return str(self._target.slurm_bin / name)

    def ping(self) -> str:
        """Prove the transport and Slurm client work: the ``sbatch --version`` text."""
        completed = self._transport.run((self._command("sbatch"), "--version"))
        _require_ok(completed, "sbatch --version", quiet=False)
        return completed.stdout.decode("utf-8", "replace").strip()

    def submit(self, argv: Sequence[str], script: bytes) -> int:
        """Submit ``script`` with the rendered ``sbatch`` argv and return the accepted job.

        Any failure raises ``Unavailable``; the caller must treat acceptance as unresolved.
        """
        completed = self._transport.run(argv, stdin=script)
        _require_ok(completed, "sbatch", quiet=False)
        try:
            return parse_receipt(completed.stdout)
        except EvidenceConflict:
            raise Unavailable("sbatch did not print exactly one job number") from None

    def test_only(self, argv: Sequence[str], script: bytes) -> tuple[bool, str, str]:
        """Ask ``sbatch --test-only`` whether Slurm would accept this request now."""
        completed = self._transport.run((*argv, "--test-only"), stdin=script)
        return (
            completed.returncode == 0,
            completed.stdout.decode("utf-8", "replace").rstrip("\n"),
            completed.stderr.decode("utf-8", "replace").rstrip("\n"),
        )

    def config(self) -> dict[str, str]:
        """``scontrol show config``: each setting's name and its value as Slurm prints it."""
        argv = (self._command("scontrol"), "show", "config")
        completed = _require_ok(self._transport.run(argv), "scontrol show config", quiet=False)
        pairs = (
            line.partition("=") for line in completed.stdout.decode("utf-8", "replace").split("\n")
        )
        return {name.strip(): value.strip() for name, equals, value in pairs if equals}

    def writable_directory(self, path: PurePosixPath) -> bool:
        """Whether ``path`` is a directory this user can write to, without writing anything."""
        completed = self._transport.run((*WRITABLE_DIRECTORY, str(path)))
        if completed.returncode not in (0, 1):
            raise CommandFailed(f"checking {path} failed with {diagnostic(completed)}")
        return completed.returncode == 0

    def queue(self, allocation_ids: Sequence[str]) -> dict[str, tuple[QueueRow, ...]]:
        """Every ``squeue`` row (terminal ones included) of this user's jobs named after the
        allocations, by allocation id, at most ``MAX_QUERY_NAMES`` names per query."""
        rows: dict[str, tuple[QueueRow, ...]] = {}
        for start in range(0, len(allocation_ids), MAX_QUERY_NAMES):
            batch = allocation_ids[start : start + MAX_QUERY_NAMES]
            names = {job_name(allocation_id): allocation_id for allocation_id in batch}
            argv = (
                self._command("squeue"),
                *SQUEUE_OPTIONS,
                f"--name={','.join(names)}",
                SQUEUE_FORMAT,
            )
            found = parse_queue(_require_ok(self._transport.run(argv), "squeue").stdout, names)
            rows.update((names[name], tuple(items)) for name, items in found.items())
        return rows

    def markers(self, task_counts: Mapping[str, int]) -> dict[str, tuple[Marker, ...]]:
        """The run markers of each allocation (by id, with its Task count) in the target's
        ``log_root``, read as ``marker_patterns`` says."""
        log_root = self._target.log_root
        allocation_ids = tuple(task_counts)
        found: dict[str, list[Marker]] = {allocation_id: [] for allocation_id in allocation_ids}
        if not allocation_ids:
            return {}
        for patterns in marker_patterns(log_root, task_counts):
            completed = _require_ok(
                self._transport.run(find_argv(log_root, patterns)), f"listing {log_root}"
            )
            for allocation_id, items in parse_markers(
                completed.stdout, log_root, allocation_ids
            ).items():
                found[allocation_id].extend(items)
        return {allocation_id: tuple(items) for allocation_id, items in found.items()}

    def tail(self, path: PurePosixPath, max_bytes: int) -> tuple[bytes, bool]:
        """The last ``max_bytes`` of a remote file and whether earlier bytes exist.

        Failures raise ``Unavailable`` without any partial content or remote diagnostics.
        """
        if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_STREAM_BYTES:
            raise ConfigurationError(f"max_bytes must be an integer from 1 to {MAX_STREAM_BYTES}")
        limit = max_bytes + 1
        try:
            completed = self._transport.run(
                (TAIL, "-c", str(limit), "--", str(path)), max_stdout=limit
            )
        except Unavailable:
            raise Unavailable("log is unavailable") from None
        if completed.returncode != 0 or completed.stderr or len(completed.stdout) > limit:
            raise Unavailable("log is unavailable")
        if len(completed.stdout) == limit:
            return completed.stdout[1:], True
        return completed.stdout, False

    def cancel(self, allocation_id: str) -> None:
        """Cancel this user's jobs named after the allocation. A reused job number can never be
        hit, and a job that already ended is not an error."""
        argv = (self._command("scancel"), "--me", f"--name={job_name(allocation_id)}")
        _require_ok(self._transport.run(argv), "scancel", quiet=False)

    def order(self, lease_id: str, step_id: str, script: bytes) -> None:
        """Write one step's order into its lease's order directory, atomically. ``Unavailable``
        leaves it unknown whether the order exists."""
        directory = order_directory(self._target.log_root, lease_id)
        argv = (*ORDER, str(directory), step_id)
        _require_ok(self._transport.run(argv, stdin=script), f"ordering step {step_id}")

    def withdraw(self, lease_id: str, step_id: str, job: int, *, stop: bool) -> bool:
        """Claim a step's order for the client: ``True`` when the client's claim won, now or
        earlier, so the step never runs (its void marker records that); ``False`` when the lease
        took the order first. With ``stop``, first ask the lease to stop the step (its stop
        marker), which matters only when the lease took it."""
        log_root = self._target.log_root
        marker = log_root / f"{step_id}.{job}"
        argv = (
            *WITHDRAW,
            str(order_directory(log_root, lease_id)),
            step_id,
            f"{marker}.void",
            f"{marker}.stop" if stop else "",
        )
        completed = _require_ok(self._transport.run(argv), f"withdrawing step {step_id}")
        if completed.stdout not in (b"withdrawn\n", b"taken\n"):
            raise CommandFailed(f"withdrawing step {step_id} printed an unexpected reply")
        return completed.stdout == b"withdrawn\n"
