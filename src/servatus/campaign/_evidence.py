"""Scheduler evidence: execution states, queue rows, run markers, and their pure parsing.

Servatus observes an allocation from two sources: the job's ``squeue`` row while Slurm still lists
it, and the marker files its batch script writes to ``log_root``, read by name and mtime. A finished
job stays in ``squeue --states=all`` for ``MinJobAge`` (minutes), far longer than an NFS client
caches a directory listing, so markers read once the row is gone are complete. Run time comes from
the same markers: the file server stamps every mtime, so durations never mix clocks. The functions
here are pure; ``_scheduler`` runs the commands.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import PurePosixPath

from ..errors import EvidenceConflict
from ._config import CONTROL, TOKEN, YIELD_EXIT


class Execution(StrEnum):
    """What one Task's step did in one allocation.

    ``ENDING``: Slurm reports the job ended, but its outcome is final only once Slurm forgets the
    job (``MinJobAge`` later), when every run marker is certain to be visible.
    ``STOPPED``: the allocation was interrupted (cancel, time limit), never started the Task, or
    the step stopped to continue later (exit 75, or a nonzero exit after the pre-limit signal).
    ``LOST``: the batch shell vanished without recording anything (SIGKILL, node failure); the
    step may even still run on a node cut off from the controller.
    """

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    ENDING = "ENDING"
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
    at: datetime


@dataclass(frozen=True, slots=True)
class Stop:
    """``<allocation>.<job>.stop``: the batch script was interrupted and stopped its steps."""

    job: int
    at: datetime


@dataclass(frozen=True, slots=True)
class Limit:
    """``<allocation>.<job>.limit``: the batch shell of a link that hands off received the
    pre-limit signal and relayed it to its steps."""

    job: int
    at: datetime


@dataclass(frozen=True, slots=True)
class End:
    """``<allocation>.<job>.<slot>.<exit code>.end``: that slot's step returned."""

    job: int
    slot: int
    exit_code: int
    at: datetime


@dataclass(frozen=True, slots=True)
class Handoff:
    """``<allocation>.<slot>.go``: the previous link of the chain handed this slot on. It exists
    before this link runs, so it names no job and never proves acceptance. Its content, the
    Task's run time so far, is for the step; Servatus reads only its name and mtime."""

    slot: int
    at: datetime


Marker = Start | Stop | Limit | End | Handoff


class StopReason(StrEnum):
    """Why a ``STOPPED`` Task stopped, when the evidence says.

    ``TIME_LIMIT``: the step was stopped by, or exited nonzero after, the pre-limit signal of a
    link that hands off. ``YIELDED``: the step exited 75. ``BUDGET_SPENT``: the step exited 75 on
    the last link of a budgeted chain, which gets the signal as a plain allocation does.
    ``NEVER_STARTED``: the allocation never ran the Task.
    """

    TIME_LIMIT = "time limit"
    YIELDED = "yielded"
    BUDGET_SPENT = "budget spent"
    NEVER_STARTED = "never started"


@dataclass(frozen=True, slots=True)
class StepEnd:
    """When a slot's step returned, and its exit code."""

    exit_code: int
    at: datetime


def _clamp(value: timedelta, bound: timedelta) -> timedelta:
    return min(max(value, timedelta(0)), bound)


@dataclass(frozen=True, slots=True)
class Observation:
    """What one accepted allocation showed: its ``squeue`` state while Slurm lists the job
    (``None`` once purged), the mtimes of its start, stop, and limit markers, each slot's
    ``StepEnd`` (``None`` until its step returned), and whether each slot was ``handed`` to this
    allocation. Link 1 has every slot. A later link has a slot once its handoff marker appeared,
    unless it appeared only after this link started (a predecessor cut off from Slurm)."""

    slurm_state: str | None
    started_at: datetime | None
    stopped_at: datetime | None
    limit_at: datetime | None
    ends: tuple[StepEnd | None, ...]
    handed: tuple[bool, ...]

    @property
    def live(self) -> bool:
        """Slurm still lists the job, settling included; retry waits until it is purged."""
        return self.slurm_state is not None

    def execution(self, slot: int) -> tuple[Execution, int | None, StopReason | None]:
        """The Task in ``slot``, its exit code (steps that returned), and why it stopped."""
        state, end = self.slurm_state, self.ends[slot]
        if state is not None and _queued(state):
            return Execution.QUEUED, None, None
        if not self.handed[slot]:
            return Execution.STOPPED, None, StopReason.NEVER_STARTED
        if end is not None:
            code = end.exit_code
            if code == 0:
                return Execution.SUCCEEDED, code, None
            if code == YIELD_EXIT:
                return Execution.STOPPED, code, StopReason.YIELDED
            if self.limit_at is not None and self.limit_at <= end.at:
                return Execution.STOPPED, code, StopReason.TIME_LIMIT
            return Execution.FAILED, code, None
        if state is not None:
            return (Execution.ENDING if state in _ENDED else Execution.RUNNING), None, None
        if self.started_at is None:
            return Execution.STOPPED, None, StopReason.NEVER_STARTED
        if self.stopped_at is not None:
            reason = None if self.limit_at is None else StopReason.TIME_LIMIT
            return Execution.STOPPED, None, reason
        return Execution.LOST, None, None

    def run_time(self, slot: int, now: datetime, bound: timedelta) -> tuple[timedelta, bool]:
        """How long ``slot`` ran in this allocation, at most ``bound`` (its time limit), and
        whether that is exact: a step still running counts until ``now``, and a LOST one counts
        ``bound``, both inexact. A slot never handed or never started ran for no time."""
        started = self.started_at
        if not self.handed[slot] or started is None:
            return timedelta(0), True
        end = self.ends[slot]
        stopped = end.at if end is not None else self.stopped_at
        if stopped is not None:
            return _clamp(stopped - started, bound), True
        if self.live:
            return _clamp(now - started, bound), False
        return bound, False


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
_MARKER = re.compile(
    rf"(?:([1-9][0-9]*)\.(?:(start|stop|limit)|(0|[1-9][0-9]*)\.({_EXIT_CODE})\.end)"
    r"|(0|[1-9][0-9]*)\.go)\Z"
)
_TIMED = re.compile(rb"([0-9]{1,12})(?:\.([0-9]{1,10}))? (.+)")
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


def _mtime(seconds: bytes, fraction: bytes | None) -> datetime:
    """An exact UTC time from ``find -printf %T@`` (epoch seconds, optional fraction)."""
    micro = int((fraction or b"0")[:6].ljust(6, b"0"))
    return datetime.fromtimestamp(int(seconds), UTC) + timedelta(microseconds=micro)


def parse_markers(
    output: bytes, log_root: PurePosixPath, allocation_ids: Collection[str]
) -> dict[str, list[Marker]]:
    """Markers of the queried allocations in a ``find LOG_ROOT ... -printf '%T@ %p\\n'``
    listing: one line per file, its mtime in epoch seconds, a space, then its path.

    Files of other allocations are ignored. A line that is not ``<mtime> <path>``, or a name with
    a queried allocation prefix that breaks the marker grammar, is an ``EvidenceConflict``.
    """
    markers: dict[str, list[Marker]] = {allocation_id: [] for allocation_id in allocation_ids}
    root = str(log_root).rstrip("/").encode() + b"/"
    for line in _lines(output, None):
        timed = _TIMED.fullmatch(line)
        if timed is None:
            raise EvidenceConflict("the run marker listing is malformed")
        path = timed.group(3)
        name = path.removeprefix(root)
        match = _ALLOCATION.match(name)
        if name == path or match is None or match.group(1).decode() not in markers:
            continue
        allocation_id = match.group(1).decode()
        try:
            parsed = _MARKER.fullmatch(name[match.end() :].decode("ascii"))
        except UnicodeDecodeError:
            parsed = None
        if parsed is None:
            raise EvidenceConflict(f"allocation {allocation_id} has a malformed run marker")
        at = _mtime(timed.group(1), timed.group(2))
        job, kind, slot, code, handed = parsed.groups()
        marker: Marker
        if handed is not None:
            marker = Handoff(int(handed), at)
        elif kind == "start":
            marker = Start(int(job), at)
        elif kind == "stop":
            marker = Stop(int(job), at)
        elif kind == "limit":
            marker = Limit(int(job), at)
        else:
            marker = End(int(job), int(slot), int(code), at)
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
    *,
    gated: bool = False,
) -> Observation:
    """One accepted allocation's ``Observation``. ``gated`` allocations (chain links after the
    first) run a slot only once it is handed to them. Evidence of another job, of a slot the
    allocation does not have, or a handoff to an ungated allocation is an ``EvidenceConflict``
    naming the allocation."""
    row = queue_row(allocation_id, rows, job)
    started = stopped = limited = None
    ends: list[StepEnd | None] = [None] * task_count
    handoffs: list[datetime | None] = [None] * task_count
    for marker in markers:
        if isinstance(marker, Handoff):
            if not gated or marker.slot >= task_count:
                raise EvidenceConflict(
                    f"allocation {allocation_id}: unexpected handoff marker for slot {marker.slot}"
                )
            handoffs[marker.slot] = marker.at
        elif marker.job != job:
            raise EvidenceConflict(
                f"allocation {allocation_id}: a run marker names job {marker.job}, not the "
                f"accepted job {job}"
            )
        elif isinstance(marker, Start):
            started = marker.at
        elif isinstance(marker, Stop):
            stopped = marker.at
        elif isinstance(marker, Limit):
            limited = marker.at
        elif marker.slot >= task_count or ends[marker.slot] is not None:
            raise EvidenceConflict(
                f"allocation {allocation_id}: unexpected end marker for slot {marker.slot}"
            )
        else:
            ends[marker.slot] = StepEnd(marker.exit_code, marker.at)
    handed = tuple(
        not gated or (at is not None and (started is None or at <= started)) for at in handoffs
    )
    return Observation(
        None if row is None else row.state, started, stopped, limited, tuple(ends), handed
    )


def evidenced_job(
    allocation_id: str, rows: Sequence[QueueRow], markers: Iterable[Marker]
) -> int | None:
    """The job that queue rows or run markers prove for an unresolved allocation, if any.
    Handoff markers name no job, so they prove nothing."""
    named = {marker.job for marker in markers if not isinstance(marker, Handoff)}
    jobs = sorted({row.job for row in rows} | named)
    if len(jobs) > 1:
        shown = ", ".join(map(str, jobs))
        raise EvidenceConflict(f"allocation {allocation_id}: evidence names jobs {shown}")
    return jobs[0] if jobs else None
