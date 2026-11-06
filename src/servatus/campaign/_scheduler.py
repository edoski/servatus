"""Slurm operations over one ``Transport``: submit, test, read the queue and run markers, tail,
and cancel.

Every method issues fixed, absolute argument vectors and parses replies with the pure functions in
``_evidence``. Command failures raise ``Unavailable``; untrustworthy replies raise
``EvidenceConflict``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath

from ..errors import ConfigurationError, EvidenceConflict, Unavailable
from ._config import CONTROL
from ._evidence import Marker, QueueRow, parse_markers, parse_queue, parse_receipt
from ._remote import MAX_STREAM_BYTES, Completed, Transport
from ._script import job_name

MAX_QUERY_NAMES = 100
TAIL = "/usr/bin/tail"
FIND = "/usr/bin/find"
SQUEUE_OPTIONS = ("--me", "--all", "--states=all", "--noheader")
SQUEUE_FORMAT = "--format=%i|%j|%T"
MARKER_SUFFIXES = (".start", ".stop", ".end")
# Generous per-marker line length beyond log_root: "/", 24 hex digits, and the longest suffix.
_MARKER_LINE_BYTES = 64


def _diagnostic(completed: Completed) -> str:
    """The first stderr line, bounded and printable, for operator-facing messages."""
    first = completed.stderr.decode("utf-8", "replace").strip().partition("\n")[0]
    text = CONTROL.sub("?", first)[:200]
    return f"exit status {completed.returncode}" + (f": {text}" if text else "")


def _require_ok(completed: Completed, what: str, *, quiet: bool = True) -> Completed:
    """``completed`` if it exited 0 (and, when ``quiet``, wrote nothing to stderr)."""
    if completed.returncode != 0 or (quiet and completed.stderr):
        raise Unavailable(f"{what} failed with {_diagnostic(completed)}")
    return completed


def find_argv(log_root: PurePosixPath, prefix: str = "") -> tuple[str, ...]:
    """List the run markers in ``log_root`` whose names start with ``prefix``."""
    names: list[str] = []
    for suffix in MARKER_SUFFIXES:
        names.extend(("-o", "-name", f"{prefix}*{suffix}"))
    return (FIND, str(log_root), "-maxdepth", "1", "-type", "f", "(", *names[1:], ")")


class Scheduler:
    """Slurm commands under ``slurm_bin``, run through ``transport``."""

    def __init__(self, transport: Transport, slurm_bin: PurePosixPath | str) -> None:
        path = PurePosixPath(slurm_bin)
        if not path.is_absolute():
            raise ConfigurationError("slurm_bin must be an absolute POSIX path")
        self._transport = transport
        self._bin = path

    def _command(self, name: str) -> str:
        return str(self._bin / name)

    def _sbatch_request(self, argv: Sequence[str]) -> tuple[str, ...]:
        fields = tuple(argv)
        if not fields or fields[0] != self._command("sbatch") or "--test-only" in fields:
            raise ConfigurationError("submission argv must start with this target's sbatch")
        return fields

    def ping(self) -> str:
        """Prove the transport and Slurm client work: the ``sbatch --version`` text."""
        completed = self._transport.run((self._command("sbatch"), "--version"))
        _require_ok(completed, "sbatch --version", quiet=False)
        return completed.stdout.decode("utf-8", "replace").strip()

    def submit(self, argv: Sequence[str], script: bytes) -> int:
        """Submit ``script`` with the rendered ``sbatch`` argv and return the accepted job.

        Any failure raises ``Unavailable``; the caller must treat acceptance as unresolved.
        """
        completed = self._transport.run(self._sbatch_request(argv), stdin=script)
        _require_ok(completed, "sbatch", quiet=False)
        try:
            return parse_receipt(completed.stdout)
        except EvidenceConflict:
            raise Unavailable("sbatch did not print exactly one job number") from None

    def test_only(self, argv: Sequence[str], script: bytes) -> tuple[bool, str, str]:
        """Ask ``sbatch --test-only`` whether Slurm would accept this request now."""
        completed = self._transport.run((*self._sbatch_request(argv), "--test-only"), stdin=script)
        return (
            completed.returncode == 0,
            completed.stdout.decode("utf-8", "replace").rstrip("\n"),
            completed.stderr.decode("utf-8", "replace").rstrip("\n"),
        )

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

    def markers(
        self, log_root: PurePosixPath, task_counts: Mapping[str, int]
    ) -> dict[str, tuple[Marker, ...]]:
        """The run markers of each allocation (by id, with its Task count) in ``log_root``.

        One directory read, split by the allocations' leading hex digit only when their own
        markers could exceed the transport's stream bound.
        """
        allocation_ids = tuple(task_counts)
        estimate = sum(
            (3 + count) * (len(str(log_root)) + _MARKER_LINE_BYTES)
            for count in task_counts.values()
        )
        prefixes = [""]
        if estimate > MAX_STREAM_BYTES // 2:
            prefixes = sorted({allocation_id[0] for allocation_id in allocation_ids})
        found: dict[str, list[Marker]] = {allocation_id: [] for allocation_id in allocation_ids}
        for prefix in prefixes:
            completed = _require_ok(
                self._transport.run(find_argv(log_root, prefix)), f"listing {log_root}"
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
        text = str(path)
        if not PurePosixPath(text).is_absolute() or CONTROL.search(text):
            raise ConfigurationError("log path must be an absolute POSIX path")
        limit = max_bytes + 1
        try:
            completed = self._transport.run((TAIL, "-c", str(limit), "--", text), max_stdout=limit)
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
