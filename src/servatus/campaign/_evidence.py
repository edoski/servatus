"""Scheduler evidence: execution states, queue rows, run markers, and their pure parsing.

Servatus observes an allocation from two sources: the job's ``squeue`` row while Slurm still lists
it, and the empty marker files its batch script writes to ``log_root``. A finished job stays in
``squeue --states=all`` for ``MinJobAge`` (minutes), far longer than an NFS client caches a
directory listing, so markers read once the row is gone are complete. The functions here are pure;
``_scheduler`` runs the commands.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

from ..errors import EvidenceConflict
from ._config import CONTROL, TOKEN


class Execution(StrEnum):
    """What one Task's step did in one allocation.

    ``STOPPED``: the allocation was interrupted (cancel, time limit) or never started the Task.
    ``LOST``: the batch shell vanished without recording anything (SIGKILL, node failure); the
    step may even still run on a node cut off from the controller.
    """

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"
    LOST = "LOST"

    @property
    def terminal(self) -> bool:
        """Finished with a known outcome. ``LOST`` is not: nothing recorded how it ended."""
        return self in (Execution.SUCCEEDED, Execution.FAILED, Execution.STOPPED)


_RUNNING = frozenset({"RUNNING", "COMPLETING", "SIGNALING", "STAGE_OUT", "SUSPENDED", "STOPPED"})
_ENDED = frozenset(
    {
        "COMPLETED",
        "CANCELLED",
        "FAILED",
        "TIMEOUT",
        "NODE_FAIL",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
        "OUT_OF_MEMORY",
    }
)


def _queued(state: str) -> bool:
    """Neither in the RUNNING family nor ended: pending, held, or a keyword Servatus does not
    know, so nothing may have started."""
    return state not in _RUNNING and state not in _ENDED


@dataclass(frozen=True, slots=True)
class QueueRow:
    """One ``squeue`` row of a job carrying an allocation identity. Every row is live: a job in
    a terminal state is still settling until Slurm purges it."""

    job: int
    state: str

    @property
    def settling(self) -> bool:
        return self.state in _ENDED


@dataclass(frozen=True, slots=True)
class Start:
    """``<allocation>.<job>.start``: the batch script began; without it no Task process ran."""

    job: int


@dataclass(frozen=True, slots=True)
class Stop:
    """``<allocation>.<job>.stop``: the batch script was interrupted and stopped its steps."""

    job: int


@dataclass(frozen=True, slots=True)
class End:
    """``<allocation>.<job>.<slot>.<exit code>.end``: that slot's step returned."""

    job: int
    slot: int
    exit_code: int


Marker = Start | Stop | End


@dataclass(frozen=True, slots=True)
class Observation:
    """What one accepted allocation showed: its ``squeue`` state while Slurm lists the job
    (``None`` once purged) and its run markers, with one exit code per slot (``None`` until that
    slot's step returned)."""

    slurm_state: str | None
    started: bool
    stopped: bool
    exit_codes: tuple[int | None, ...]

    @property
    def live(self) -> bool:
        """Slurm still lists the job, settling included; retry waits until it is purged."""
        return self.slurm_state is not None

    def execution(self, slot: int) -> tuple[Execution, int | None]:
        """The Task in ``slot`` and its exit code (terminal steps only)."""
        state, code = self.slurm_state, self.exit_codes[slot]
        if state is not None and _queued(state):
            return Execution.QUEUED, None
        if code is not None:
            return (Execution.SUCCEEDED if code == 0 else Execution.FAILED), code
        if state is not None:
            return Execution.RUNNING, None
        if self.stopped or not self.started:
            return Execution.STOPPED, None
        return Execution.LOST, None


@dataclass(frozen=True, slots=True)
class Observed:
    """One observation of a Campaign: accepted allocations by id, and ``strays``, the jobs Slurm
    still lists for allocations recorded as not submitted."""

    attempts: Mapping[str, Observation]
    strays: Mapping[str, int]


# --- Parsing ---------------------------------------------------------------------------------

MAX_REPLY_LINES = 4096
_DECIMAL = re.compile(rb"[1-9][0-9]*\Z")
_RECEIPT = re.compile(rb"([1-9][0-9]*)\n?\Z")
_EXIT_CODE = r"(?:[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])"
_MARKER = re.compile(rf"([1-9][0-9]*)\.(?:(start|stop)|(0|[1-9][0-9]*)\.({_EXIT_CODE})\.end)\Z")
_ALLOCATION = re.compile(rb"([0-9a-f]{24})\.")


def parse_receipt(stdout: bytes) -> int:
    """Strictly parse ``sbatch --parsable`` output: one job number and an optional newline."""
    match = _RECEIPT.fullmatch(stdout)
    if match is None:
        raise EvidenceConflict("sbatch did not print exactly one job number")
    return int(match.group(1))


def _lines(output: bytes, bound: int | None) -> list[bytes]:
    if output and not output.endswith(b"\n"):
        raise EvidenceConflict("scheduler output is partial")
    lines = output[:-1].split(b"\n") if output else []
    if bound is not None and len(lines) > bound:
        raise EvidenceConflict("scheduler output exceeds its line bound")
    return lines


def _text(raw: bytes) -> str:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise EvidenceConflict("scheduler row is malformed") from None
    if CONTROL.search(text):
        raise EvidenceConflict("scheduler row is malformed")
    return text


def parse_queue(output: bytes, identities: Collection[str]) -> dict[str, list[QueueRow]]:
    """Rows of ``squeue --format=%i|%j|%T`` by job name; every name must be queried."""
    rows: dict[str, list[QueueRow]] = {identity: [] for identity in identities}
    for line in _lines(output, MAX_REPLY_LINES):
        fields = line.split(b"|")
        if len(fields) != 3 or _DECIMAL.fullmatch(fields[0]) is None:
            raise EvidenceConflict("scheduler row is malformed")
        name, state = _text(fields[1]), _text(fields[2])
        if TOKEN.fullmatch(state) is None:
            raise EvidenceConflict("scheduler state is malformed")
        if name not in rows:
            raise EvidenceConflict("scheduler returned a row for an unrelated job")
        rows[name].append(QueueRow(int(fields[0]), state))
    return rows


def parse_markers(
    output: bytes, log_root: PurePosixPath, allocation_ids: Collection[str]
) -> dict[str, list[Marker]]:
    """Markers of the queried allocations in a ``find LOG_ROOT -maxdepth 1 -type f`` listing.

    Files of other allocations are ignored. A name with a queried allocation prefix that breaks
    the marker grammar is an ``EvidenceConflict`` naming that allocation.
    """
    markers: dict[str, list[Marker]] = {allocation_id: [] for allocation_id in allocation_ids}
    root = str(log_root).rstrip("/").encode() + b"/"
    for line in _lines(output, None):
        name = line.removeprefix(root)
        match = _ALLOCATION.match(name)
        if name == line or match is None or match.group(1).decode() not in markers:
            continue
        allocation_id = match.group(1).decode()
        try:
            parsed = _MARKER.fullmatch(name[match.end() :].decode("ascii"))
        except UnicodeDecodeError:
            parsed = None
        if parsed is None:
            raise EvidenceConflict(f"allocation {allocation_id} has a malformed run marker")
        job, kind, slot, code = parsed.groups()
        marker: Marker
        if kind == "start":
            marker = Start(int(job))
        elif kind == "stop":
            marker = Stop(int(job))
        else:
            marker = End(int(job), int(slot), int(code))
        markers[allocation_id].append(marker)
    return markers


# --- Decisions -------------------------------------------------------------------------------


def queue_row(allocation_id: str, rows: Sequence[QueueRow], job: int | None) -> QueueRow | None:
    """The one row Slurm lists for an allocation, which must be ``job`` when one is recorded."""
    jobs = sorted({row.job for row in rows})
    if len(jobs) > 1:
        shown = ", ".join(map(str, jobs))
        raise EvidenceConflict(f"allocation {allocation_id}: Slurm lists jobs {shown}")
    if len(rows) > 1:
        raise EvidenceConflict(f"allocation {allocation_id}: Slurm lists job {jobs[0]} twice")
    if rows and job is not None and rows[0].job != job:
        raise EvidenceConflict(
            f"allocation {allocation_id}: Slurm lists job {rows[0].job} under its identity, "
            f"not the accepted job {job}"
        )
    return rows[0] if rows else None


def observe(
    allocation_id: str,
    job: int,
    task_count: int,
    rows: Sequence[QueueRow],
    markers: Iterable[Marker],
) -> Observation:
    """One accepted allocation's ``Observation``; evidence of another job, or of a slot it does
    not have, is an ``EvidenceConflict`` naming the allocation."""
    row = queue_row(allocation_id, rows, job)
    started = stopped = False
    exit_codes: list[int | None] = [None] * task_count
    for marker in markers:
        if marker.job != job:
            raise EvidenceConflict(
                f"allocation {allocation_id}: a run marker names job {marker.job}, not the "
                f"accepted job {job}"
            )
        if isinstance(marker, Start):
            started = True
        elif isinstance(marker, Stop):
            stopped = True
        elif marker.slot >= task_count or exit_codes[marker.slot] is not None:
            raise EvidenceConflict(
                f"allocation {allocation_id}: unexpected end marker for slot {marker.slot}"
            )
        else:
            exit_codes[marker.slot] = marker.exit_code
    return Observation(None if row is None else row.state, started, stopped, tuple(exit_codes))


def evidenced_job(
    allocation_id: str, rows: Sequence[QueueRow], markers: Iterable[Marker]
) -> int | None:
    """The job that queue rows or run markers prove for an unresolved allocation, if any."""
    jobs = sorted({row.job for row in rows} | {marker.job for marker in markers})
    if len(jobs) > 1:
        shown = ", ".join(map(str, jobs))
        raise EvidenceConflict(f"allocation {allocation_id}: evidence names jobs {shown}")
    return jobs[0] if jobs else None
