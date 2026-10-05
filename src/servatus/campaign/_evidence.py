"""Scheduler evidence: normalized states, job identities, and pure Slurm output parsing.

The type definitions at the top are the shared vocabulary of the campaign layer. Parsing and
normalization functions below them are pure; ``_scheduler`` performs the remote calls.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from ..errors import ConfigurationError, EvidenceConflict, ReconciliationError
from ._config import TOKEN


class AllocationState(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"

    @property
    def active(self) -> bool:
        return self in (AllocationState.QUEUED, AllocationState.RUNNING)

    @property
    def terminal(self) -> bool:
        return self in (
            AllocationState.SUCCEEDED,
            AllocationState.FAILED,
            AllocationState.CANCELLED,
        )


@dataclass(frozen=True, slots=True)
class JobRef:
    """A positive Slurm job number and optional federation cluster name."""

    job_id: int
    cluster: str | None = None

    def __post_init__(self) -> None:
        if type(self.job_id) is not int or self.job_id < 1:
            raise ConfigurationError("job_id must be a positive integer")
        if self.cluster is not None and (
            not isinstance(self.cluster, str) or TOKEN.fullmatch(self.cluster) is None
        ):
            raise ConfigurationError("cluster must be one safe site token")

    def __str__(self) -> str:
        return f"{self.job_id}" if self.cluster is None else f"{self.job_id};{self.cluster}"


@dataclass(frozen=True, slots=True)
class SchedulerEvidence:
    """Allocation-level evidence combined from squeue and anchored sacct history.

    ``retained`` means the scheduler still holds the work (queued, running, held, requeued) and
    blocks retry. ``problem`` is set when this allocation's evidence was contradictory; the
    state is then ``UNKNOWN`` and ``retained`` is true, which withholds only its Tasks.
    """

    state: AllocationState
    raw_state: str | None = None
    accounting_state: str | None = None
    exit_code: str | None = None
    reason: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    retained: bool = False
    problem: str | None = None


@dataclass(frozen=True, slots=True)
class StepEvidence:
    """Per-Task evidence from that Task's own ``srun`` step."""

    state: AllocationState
    raw_state: str
    exit_code: str | None = None


@dataclass(frozen=True, slots=True)
class Observation:
    """Everything observed for one accepted allocation.

    ``steps`` has one entry per slot (``None`` when that Task's step was not found), or is empty
    when the step query itself failed, so step evidence is unavailable rather than absent.
    """

    allocation: SchedulerEvidence
    steps: tuple[StepEvidence | None, ...] = ()


# --- Pure Slurm output parsing ---------------------------------------------------------------
#
# Reply-level defects (partial or oversized output, malformed rows, rows for job numbers that were
# not queried, rows from another cluster) raise ``EvidenceConflict`` for the whole query.
# Contradictions confined to one allocation's own job number become that allocation's ``problem``
# instead, so one odd job cannot hide the evidence of every other job in its batch. That includes
# a queried job number now held by a foreign job (a reused number with a different identity).

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"
WINDOW = timedelta(hours=1)
MAX_REPLY_LINES = 4096
MAX_FIELD_BYTES = 4096
MISSING_JOBS_STDERR = b"slurm_load_jobs error: Invalid job id specified\n"
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_DECIMAL = re.compile(r"[1-9][0-9]*\Z")
_STEP_ID = re.compile(r"([1-9][0-9]*)\.[0-9]+\Z")
_SLOT = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_EXIT_CODE = re.compile(r"[0-9]+:[0-9]+\Z")
_RECEIPT = re.compile(rb"([1-9][0-9]*)(?:;([A-Za-z0-9][A-Za-z0-9._-]*))?\n?\Z")
_NULL = frozenset({"", "None", "Unknown", "N/A"})
FOREIGN_JOB = "job number is held by a foreign job"

_STATES: Mapping[str, AllocationState] = {
    **dict.fromkeys(
        (
            "PENDING",
            "CONFIGURING",
            "REQUEUED",
            "RESIZING",
            "REQUEUE_HOLD",
            "REQUEUE_FED",
            "RESV_DEL_HOLD",
            "SPECIAL_EXIT",
            "EXPEDITING",
        ),
        AllocationState.QUEUED,
    ),
    **dict.fromkeys(
        ("RUNNING", "COMPLETING", "SIGNALING", "STAGE_OUT", "SUSPENDED", "STOPPED"),
        AllocationState.RUNNING,
    ),
    "COMPLETED": AllocationState.SUCCEEDED,
    "CANCELLED": AllocationState.CANCELLED,
    "PREEMPTED": AllocationState.CANCELLED,
    **dict.fromkeys(
        ("BOOT_FAIL", "DEADLINE", "FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "TIMEOUT"),
        AllocationState.FAILED,
    ),
}
_RANK = {
    AllocationState.QUEUED: 1,
    AllocationState.RUNNING: 2,
    AllocationState.SUCCEEDED: 3,
    AllocationState.FAILED: 3,
    AllocationState.CANCELLED: 3,
}


def state_base(raw: str) -> str | None:
    """The state keyword of ``CANCELLED by 1234`` or ``CANCELLED+``; None when malformed."""
    base = raw.split(" ", 1)[0].removesuffix("+")
    return base if TOKEN.fullmatch(base) is not None else None


def normalize_state(raw: str) -> AllocationState:
    """Map one Slurm state to Servatus vocabulary; anything unlisted is ``UNKNOWN``."""
    base = state_base(raw)
    return AllocationState.UNKNOWN if base is None else _STATES.get(base, AllocationState.UNKNOWN)


def format_timestamp(moment: datetime) -> str:
    """A UTC instant in the exact text form Slurm prints under ``TZ=UTC``."""
    if moment.utcoffset() is None:
        raise ConfigurationError("scheduler times must be timezone-aware")
    return moment.astimezone(UTC).strftime(TIMESTAMP_FORMAT)


def submission_window(intent_at: datetime) -> tuple[str, str]:
    """The span (intent +/- 1 h) in which an allocation's original submission must fall."""
    return format_timestamp(intent_at - WINDOW), format_timestamp(intent_at + WINDOW)


def parse_receipt(stdout: bytes) -> JobRef:
    """Strictly parse ``sbatch --parsable`` output: ``<job id>[;<cluster>]``."""
    match = _RECEIPT.fullmatch(stdout)
    if match is None:
        raise EvidenceConflict("sbatch did not print exactly one job identity")
    cluster = match.group(2)
    return JobRef(int(match.group(1)), None if cluster is None else cluster.decode("ascii"))


# --- Rows ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Expectation:
    """What one accepted allocation's rows must match within a query batch."""

    allocation_id: str
    window_start: str
    window_end: str
    task_count: int

    @property
    def identity(self) -> str:
        return f"servatus-{self.allocation_id}"


@dataclass(frozen=True, slots=True)
class ActiveRow:
    """One ``squeue`` row for a queried job."""

    submitted_at: str
    state: str
    reason: str | None
    started_at: str | None
    ended_at: str | None
    foreign: bool = False


@dataclass(frozen=True, slots=True)
class AccountingRow:
    """One ``sacct --allocations --duplicates`` row: one incarnation of a queried job."""

    cluster: str | None
    submitted_at: str
    state: str
    exit_code: str | None
    reason: str | None
    started_at: str | None
    ended_at: str | None
    foreign: bool = False


def reply_lines(output: bytes) -> tuple[bytes, ...]:
    """Newline-terminated lines of one reply; partial or oversized output is malformed."""
    if output and not output.endswith(b"\n"):
        raise EvidenceConflict("scheduler output is partial")
    lines = output[:-1].split(b"\n") if output else []
    if len(lines) > MAX_REPLY_LINES:
        raise EvidenceConflict("scheduler output exceeds its line bound")
    return tuple(lines)


def reply_fields(line: bytes, count: int) -> tuple[str, ...]:
    """Split one ``|``-separated row into exactly ``count`` strict UTF-8 fields."""
    raw = line.split(b"|")
    if len(raw) != count or any(len(field) > MAX_FIELD_BYTES for field in raw):
        raise EvidenceConflict("scheduler row is malformed")
    try:
        fields = tuple(field.decode("utf-8") for field in raw)
    except UnicodeDecodeError:
        raise EvidenceConflict("scheduler row is malformed") from None
    if any(_CONTROL.search(field) for field in fields):
        raise EvidenceConflict("scheduler row is malformed")
    return fields


def _optional(value: str) -> str | None:
    value = value.strip(" ")
    return None if value in _NULL else value


def _timestamp(value: str) -> str | None:
    value = value.strip(" ")
    if value in _NULL:
        return None
    try:
        parsed = datetime.strptime(value, TIMESTAMP_FORMAT)
    except ValueError:
        raise EvidenceConflict("scheduler timestamp is malformed") from None
    if parsed.strftime(TIMESTAMP_FORMAT) != value:
        raise EvidenceConflict("scheduler timestamp is malformed")
    return value


def _required_timestamp(value: str) -> str:
    timestamp = _timestamp(value)
    if timestamp is None:
        raise EvidenceConflict("scheduler submit time is missing")
    return timestamp


def _exit_code(value: str) -> str | None:
    value = value.strip(" ")
    if value in _NULL:
        return None
    if _EXIT_CODE.fullmatch(value) is None:
        raise EvidenceConflict("scheduler exit code is malformed")
    return value


def _state(value: str) -> str:
    value = value.strip(" ")
    if state_base(value) is None:
        raise EvidenceConflict("scheduler state is malformed")
    return value


def _job_number(value: str, expected: Mapping[int, Expectation]) -> int:
    if _DECIMAL.fullmatch(value) is None or int(value) not in expected:
        raise EvidenceConflict("scheduler returned evidence for an unrelated job")
    return int(value)


def _without_cluster_banner(lines: tuple[bytes, ...], cluster: str | None) -> tuple[bytes, ...]:
    # ``squeue --clusters`` prints a ``CLUSTER: <name>`` line even under ``--noheader``.
    if cluster is not None and lines[:1] == (f"CLUSTER: {cluster}".encode(),):
        return lines[1:]
    return lines


def is_missing_reply(returncode: int, stdout: bytes, stderr: bytes, cluster: str | None) -> bool:
    """True only for squeue's exact native reply when none of the queried jobs is active."""
    if returncode != 1 or stderr != MISSING_JOBS_STDERR:
        return False
    try:
        return _without_cluster_banner(reply_lines(stdout), cluster) == ()
    except EvidenceConflict:
        return False


def parse_active(
    output: bytes, expected: Mapping[int, Expectation], cluster: str | None
) -> dict[int, list[ActiveRow]]:
    """Parse ``squeue --format=%i|%j|%k|%V|%T|%r|%S|%e`` rows by job number.

    Queue rows must carry the exact allocation identity as both name and comment; a row that does
    not is marked ``foreign`` (the job number was reused) and confined to its allocation.
    """
    rows: dict[int, list[ActiveRow]] = {job: [] for job in expected}
    for line in _without_cluster_banner(reply_lines(output), cluster):
        job_id, name, comment, submitted, state, reason, started, ended = reply_fields(line, 8)
        job = _job_number(job_id, expected)
        identity = expected[job].identity
        rows[job].append(
            ActiveRow(
                _required_timestamp(submitted),
                _state(state),
                _optional(reason),
                _timestamp(started),
                _timestamp(ended),
                foreign=name != identity or comment != identity,
            )
        )
    return rows


def parse_accounting(
    output: bytes, expected: Mapping[int, Expectation], cluster: str | None
) -> dict[int, list[AccountingRow]]:
    """Parse ``sacct`` allocation rows (JobIDRaw, Cluster, JobName, Comment, Submit, State,
    ExitCode, Reason, Start, End) by job number, in reply order.

    The name must be the exact allocation identity; sites may omit the comment from accounting.
    A row with another identity is marked ``foreign`` and confined to its allocation.
    """
    rows: dict[int, list[AccountingRow]] = {job: [] for job in expected}
    for line in reply_lines(output):
        fields = reply_fields(line, 10)
        job_id, row_cluster, name, comment = fields[:4]
        submitted, state, exit_code, reason, started, ended = fields[4:]
        job = _job_number(job_id, expected)
        identity = expected[job].identity
        foreign = name != identity or (comment != identity and comment not in _NULL)
        found = None if row_cluster in _NULL else row_cluster
        if (found is not None and TOKEN.fullmatch(found) is None) or (
            cluster is not None and found != cluster
        ):
            raise EvidenceConflict("scheduler returned evidence from an unrelated cluster")
        rows[job].append(
            AccountingRow(
                found,
                _required_timestamp(submitted),
                _state(state),
                _exit_code(exit_code),
                _optional(reason),
                _timestamp(started),
                _timestamp(ended),
                foreign=foreign,
            )
        )
    return rows


# --- Combination -----------------------------------------------------------------------------


def _problem(
    message: str, queue: ActiveRow | None, latest: AccountingRow | None
) -> SchedulerEvidence:
    return SchedulerEvidence(
        AllocationState.UNKNOWN,
        raw_state=None if queue is None else queue.state,
        accounting_state=None if latest is None else latest.state,
        retained=True,
        problem=message,
    )


def _unfinished(state: str) -> bool:
    return not normalize_state(state).terminal


def combine(
    expectation: Expectation, active: Sequence[ActiveRow], history: Sequence[AccountingRow]
) -> SchedulerEvidence:
    """Combine one allocation's queue rows and accounting history into one evidence value.

    Accounting history must start inside the submission window; later requeue incarnations need
    strictly increasing submit times on one cluster. Within one incarnation the more advanced
    state wins (terminal over running over queued). Any unfinished or unlisted sample keeps the
    allocation ``retained``. A terminal state needs anchored accounting: a terminal queue row
    alone is ``UNKNOWN``. Contradictions, and rows from a foreign job holding this job number,
    yield ``UNKNOWN`` with ``problem`` set.
    """
    if any(row.foreign for row in active) or any(row.foreign for row in history):
        return SchedulerEvidence(AllocationState.UNKNOWN, retained=True, problem=FOREIGN_JOB)
    latest = history[-1] if history else None
    queue = active[0] if active else None
    if len(active) > 1:
        return _problem("squeue returned several rows for one job", queue, latest)
    if history:
        anchor = history[0]
        if not expectation.window_start <= anchor.submitted_at <= expectation.window_end:
            return _problem(
                "accounting history starts outside the submission window", queue, latest
            )
        for earlier, later in itertools.pairwise(history):
            if later.submitted_at <= earlier.submitted_at or later.cluster != anchor.cluster:
                return _problem("accounting requeue history is not ordered", queue, latest)
    if queue is not None and (
        queue.submitted_at < expectation.window_start
        or (history and queue.submitted_at < history[0].submitted_at)
    ):
        return _problem("queue row predates the allocation's submission", queue, latest)
    retained = (queue is not None and _unfinished(queue.state)) or (
        latest is not None and _unfinished(latest.state)
    )
    if latest is None:
        if queue is None:
            return SchedulerEvidence(AllocationState.UNKNOWN)
        state = normalize_state(queue.state)
        return SchedulerEvidence(
            AllocationState.UNKNOWN if state.terminal else state,
            raw_state=queue.state,
            reason=queue.reason,
            started_at=queue.started_at,
            ended_at=queue.ended_at,
            retained=retained,
        )
    primary: ActiveRow | AccountingRow = latest
    if queue is not None and queue.submitted_at > latest.submitted_at:
        primary = queue  # accounting has not recorded the newest incarnation yet
    elif queue is not None and queue.submitted_at == latest.submitted_at:
        chosen = _same_incarnation(queue, latest)
        if chosen is None:
            message = f"squeue reports {queue.state} but sacct reports {latest.state}"
            return _problem(message, queue, latest)
        primary = chosen
    # Otherwise the queue row was sampled before a requeue that accounting already shows.
    return SchedulerEvidence(
        normalize_state(primary.state),
        raw_state=primary.state,
        accounting_state=latest.state,
        exit_code=latest.exit_code,
        reason=primary.reason or latest.reason,
        started_at=primary.started_at or latest.started_at,
        ended_at=primary.ended_at or latest.ended_at,
        retained=retained,
    )


def _same_incarnation(queue: ActiveRow, latest: AccountingRow) -> ActiveRow | AccountingRow | None:
    if state_base(queue.state) == state_base(latest.state):
        return queue
    queue_state, accounting_state = normalize_state(queue.state), normalize_state(latest.state)
    if AllocationState.UNKNOWN in (queue_state, accounting_state):
        return None
    queue_rank, accounting_rank = _RANK[queue_state], _RANK[accounting_state]
    if queue_rank != accounting_rank:
        return queue if queue_rank > accounting_rank else latest
    if queue_state is not accounting_state:
        return None  # two different terminal outcomes
    return latest if queue_state.terminal else queue


# --- Steps -----------------------------------------------------------------------------------


def parse_steps(
    output: bytes, expected: Mapping[int, Expectation]
) -> dict[int, tuple[StepEvidence | None, ...]]:
    """Map ``sacct`` step rows (JobIDRaw, JobName, State, ExitCode) to per-slot evidence.

    Only rows named ``servatus-<allocation id>-<slot>`` under a queried job count; batch, extern,
    allocation, and foreign rows are ignored. A slot reported twice has no evidence.
    """
    found: dict[int, dict[int, list[StepEvidence]]] = {job: {} for job in expected}
    for line in reply_lines(output):
        job_step, name, state, exit_code = reply_fields(line, 4)
        match = _STEP_ID.fullmatch(job_step)
        if match is None or (job := int(match.group(1))) not in expected:
            continue
        expectation = expected[job]
        slot_text = name.removeprefix(f"{expectation.identity}-")
        if slot_text == name or _SLOT.fullmatch(slot_text) is None:
            continue
        slot = int(slot_text)
        if slot < expectation.task_count:
            raw = _state(state)
            evidence = StepEvidence(normalize_state(raw), raw, _exit_code(exit_code))
            found[job].setdefault(slot, []).append(evidence)
    return {
        job: tuple(
            items[0] if len(items := found[job].get(slot, [])) == 1 else None
            for slot in range(expectation.task_count)
        )
        for job, expectation in expected.items()
    }


# --- Identity --------------------------------------------------------------------------------


def parse_identity(squeue: bytes, sacct: bytes, identity: str) -> tuple[JobRef, ...]:
    """Every job carrying ``identity`` in ``squeue --format=%i|%j|%k`` and ``sacct
    --format=JobIDRaw,JobName,Comment,Cluster`` replies, in job-number order.

    An empty result proves absence. Malformed or unrelated rows, and one job number reported on
    several clusters, raise ``ReconciliationError``: such replies prove nothing.
    """
    candidates: dict[int, set[str | None]] = {}
    try:
        for line in reply_lines(squeue):
            job_id, name, comment = reply_fields(line, 3)
            if name != identity or comment != identity:
                raise ReconciliationError("scheduler returned an unrelated allocation identity")
            candidates.setdefault(_candidate(job_id), set())
        for line in reply_lines(sacct):
            job_id, name, comment, cluster = reply_fields(line, 4)
            if name != identity or (comment != identity and comment not in _NULL):
                raise ReconciliationError("scheduler returned an unrelated allocation identity")
            if cluster and TOKEN.fullmatch(cluster) is None:
                raise ReconciliationError("scheduler returned an invalid cluster identity")
            candidates.setdefault(_candidate(job_id), set()).add(cluster or None)
    except EvidenceConflict as error:
        raise ReconciliationError(f"scheduler identity evidence is malformed: {error}") from None
    jobs: list[JobRef] = []
    for job_id, clusters in sorted(candidates.items()):
        known = {cluster for cluster in clusters if cluster is not None}
        if len(known) > 1:
            raise ReconciliationError("scheduler returned conflicting cluster identities")
        jobs.append(JobRef(job_id, next(iter(known), None)))
    return tuple(jobs)


def _candidate(value: str) -> int:
    if _DECIMAL.fullmatch(value) is None:
        raise ReconciliationError("scheduler returned an invalid job identity")
    return int(value)
