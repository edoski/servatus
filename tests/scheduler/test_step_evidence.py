"""A step's evidence: its own markers read through its lease's queue row and markers."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import PurePosixPath

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scheduler_fixtures import ALLOCATION, AT, LOG_ROOT, allocation, listing

from servatus.campaign._evidence import (
    End,
    Execution,
    Handoff,
    Limit,
    Marker,
    Observation,
    QueueRow,
    Start,
    StepEnd,
    Stop,
    StopReason,
    Void,
    observe,
    observe_step,
    parse_markers,
)
from servatus.errors import EvidenceConflict

LEASE = ALLOCATION
STEP = allocation(7)
JOB = 42
BOUND = timedelta(minutes=30)


def at(minutes: float) -> timedelta:
    return timedelta(minutes=minutes)


def step(
    state: str | None, own: Sequence[Marker] = (), lease: Sequence[Marker] = ()
) -> Observation:
    rows = [] if state is None else [QueueRow(JOB, state)]
    return observe_step(STEP, LEASE, JOB, rows, own, lease)


def start(minutes: float = 1) -> Start:
    return Start(JOB, AT + at(minutes))


def stop(minutes: float) -> Stop:
    return Stop(JOB, AT + at(minutes))


def limit(minutes: float) -> Limit:
    return Limit(JOB, AT + at(minutes))


def end(code: int, minutes: float = 10, slot: int = 0) -> End:
    return End(JOB, slot, code, AT + at(minutes))


def void(minutes: float = 0) -> Void:
    return Void(JOB, AT + at(minutes))


HELD = (Start(JOB, AT),)


# --- Parsing ---------------------------------------------------------------------------------


def test_a_void_marker_parses() -> None:
    markers = parse_markers(listing(f"{STEP}.42.void"), PurePosixPath(LOG_ROOT), [STEP])
    assert markers == {STEP: [Void(42, AT)]}


def test_an_allocation_with_a_void_marker_is_a_conflict() -> None:
    with pytest.raises(EvidenceConflict, match=f"allocation {ALLOCATION}: unexpected void"):
        observe(ALLOCATION, JOB, 1, [], [Start(JOB, AT), Void(JOB, AT)])


# --- Execution -------------------------------------------------------------------------------

Outcome = tuple[Execution, int | None, StopReason | None]
NEVER = (Execution.STOPPED, None, StopReason.NEVER_STARTED)

EXECUTIONS: list[tuple[str | None, Sequence[Marker], Sequence[Marker], Outcome]] = [
    ("PENDING", (), (), (Execution.QUEUED, None, None)),
    ("PENDING", (void(),), (), NEVER),
    ("RUNNING", (void(),), HELD, NEVER),
    (None, (void(),), HELD, NEVER),
    ("RUNNING", (start(), end(0)), HELD, (Execution.SUCCEEDED, 0, None)),
    (None, (start(), end(75)), HELD, (Execution.STOPPED, 75, StopReason.YIELDED)),
    ("RUNNING", (start(), end(1)), HELD, (Execution.FAILED, 1, None)),
    (
        "RUNNING",
        (start(), end(1)),
        (*HELD, limit(5)),
        (Execution.STOPPED, 1, StopReason.TIME_LIMIT),
    ),
    (
        None,
        (start(), end(138)),
        (*HELD, limit(10)),
        (Execution.STOPPED, 138, StopReason.TIME_LIMIT),
    ),
    ("RUNNING", (start(), end(1)), (*HELD, limit(11)), (Execution.FAILED, 1, None)),
    ("RUNNING", (start(), stop(5), end(143)), HELD, (Execution.STOPPED, 143, None)),
    ("RUNNING", (start(), stop(10), end(143)), HELD, (Execution.STOPPED, 143, None)),
    ("RUNNING", (stop(5), end(143)), HELD, (Execution.STOPPED, 143, StopReason.NEVER_STARTED)),
    (None, (start(), end(143)), (*HELD, stop(5)), (Execution.STOPPED, 143, None)),
    ("RUNNING", (start(), stop(11), end(1)), HELD, (Execution.FAILED, 1, None)),
    ("RUNNING", (), HELD, (Execution.QUEUED, None, None)),
    ("COMPLETING", (start(),), HELD, (Execution.RUNNING, None, None)),
    ("RUNNING", (start(), stop(5)), HELD, (Execution.RUNNING, None, None)),
    ("COMPLETED", (start(),), HELD, (Execution.ENDING, None, None)),
    ("TIMEOUT", (), HELD, (Execution.ENDING, None, None)),
    (None, (), HELD, NEVER),
    (None, (stop(5),), HELD, NEVER),
    (None, (start(), stop(5)), HELD, (Execution.STOPPED, None, None)),
    (None, (start(),), (*HELD, stop(5)), (Execution.STOPPED, None, None)),
    (
        None,
        (start(), stop(5)),
        (*HELD, limit(3)),
        (Execution.STOPPED, None, StopReason.TIME_LIMIT),
    ),
    (None, (start(),), HELD, (Execution.LOST, None, None)),
    (None, (start(),), (*HELD, limit(3)), (Execution.LOST, None, None)),
]


@pytest.mark.parametrize(("state", "own", "lease", "expected"), EXECUTIONS)
def test_a_step_follows_the_step_execution_table(
    state: str | None, own: Sequence[Marker], lease: Sequence[Marker], expected: Outcome
) -> None:
    assert step(state, own, lease).execution(0) == expected


def test_a_step_takes_the_lease_row_and_limit_and_its_own_start_and_end() -> None:
    own = (start(1), stop(15), end(143, 18))
    lease = (Start(JOB, AT), stop(20), limit(5))
    assert step("RUNNING", own, lease) == Observation(
        "RUNNING",
        AT + at(1),
        AT + at(15),
        AT + at(5),
        (StepEnd(143, AT + at(18)),),
        (True,),
        step=True,
    )


@pytest.mark.parametrize(
    ("requested", "held", "expected"),
    [(5, 20, 5), (20, 5, 5), (None, 5, 5), (5, None, 5), (None, None, None)],
)
def test_a_step_stopped_at_the_earlier_of_its_stop_request_and_the_lease_stop(
    requested: float | None, held: float | None, expected: float | None
) -> None:
    own = () if requested is None else (stop(requested),)
    lease = HELD if held is None else (*HELD, stop(held))
    stopped = step("RUNNING", own, lease).stopped_at
    assert stopped == (None if expected is None else AT + at(expected))


@pytest.mark.parametrize(
    ("state", "own", "live"),
    [
        ("PENDING", (), True),
        ("RUNNING", (), True),
        ("RUNNING", (start(),), True),
        ("COMPLETED", (start(),), True),
        ("RUNNING", (start(), end(0)), False),
        ("RUNNING", (start(), end(1)), False),
        ("RUNNING", (void(),), False),
        (None, (start(),), False),
        (None, (), False),
    ],
)
def test_a_step_is_live_while_its_lease_is_listed_and_it_neither_ended_nor_was_withdrawn(
    state: str | None, own: Sequence[Marker], live: bool
) -> None:
    assert step(state, own, HELD).live is live


@pytest.mark.parametrize(
    ("state", "ended"),
    [
        (None, True),
        ("COMPLETED", True),
        ("NODE_FAIL", True),
        ("RUNNING", False),
        ("PENDING", False),
    ],
)
def test_a_step_ended_once_its_lease_is_gone_or_settling(state: str | None, ended: bool) -> None:
    assert step(state, (start(), end(0)), HELD).ended is ended


@pytest.mark.parametrize(
    ("state", "own", "now", "expected"),
    [
        ("RUNNING", (), 60, (timedelta(0), True)),
        ("RUNNING", (void(),), 60, (timedelta(0), True)),
        (None, (void(),), 60, (timedelta(0), True)),
        ("RUNNING", (start(1), end(0, 11)), 60, (at(10), True)),
        (None, (start(1), end(1, 21)), 60, (at(20), True)),
        ("RUNNING", (start(1),), 8, (at(7), False)),
        ("RUNNING", (start(1),), 100, (BOUND, False)),
        ("RUNNING", (start(1), stop(5)), 8, (at(7), False)),
        (None, (start(1), stop(5)), 60, (at(4), True)),
        (None, (start(1),), 60, (BOUND, False)),
    ],
    ids=[
        "never",
        "withdrawn",
        "withdrawn-gone",
        "ended",
        "failed",
        "running",
        "clamped",
        "stopping",
        "stopped",
        "lost",
    ],
)
def test_a_step_run_time_counts_from_its_own_start_marker(
    state: str | None, own: Sequence[Marker], now: float, expected: tuple[timedelta, bool]
) -> None:
    assert step(state, own, HELD).run_time(0, AT + at(now), BOUND) == expected


# --- Conflicts -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("own", "lease", "message"),
    [
        ((Start(43, AT),), (), f"step {STEP}: a run marker does not belong to job 42"),
        ((Stop(43, AT),), (), f"step {STEP}: a run marker does not belong"),
        ((End(43, 0, 0, AT),), (), f"step {STEP}: a run marker does not belong"),
        ((Void(43, AT),), (), f"step {STEP}: a run marker does not belong"),
        ((limit(1),), (), f"step {STEP}: a run marker does not belong"),
        ((Handoff(0, AT),), (), f"step {STEP}: a run marker does not belong"),
        ((end(0, slot=1),), (), f"step {STEP}: unexpected end marker for slot 1"),
        ((end(0), end(1)), (), f"step {STEP}: unexpected end marker for slot 0"),
        ((void(), start()), (), f"step {STEP} was withdrawn, yet it ran"),
        ((end(0), void()), (), f"step {STEP} was withdrawn, yet it ran"),
        ((), (end(0),), f"allocation {LEASE}: unexpected end marker for slot 0"),
        ((), (Start(43, AT),), f"allocation {LEASE}: a run marker names job 43"),
        ((), (void(),), f"allocation {LEASE}: unexpected void marker"),
    ],
)
def test_impossible_step_evidence_is_a_conflict(
    own: Sequence[Marker], lease: Sequence[Marker], message: str
) -> None:
    with pytest.raises(EvidenceConflict, match=message):
        step("RUNNING", own, lease)


def test_a_lease_row_of_another_job_is_a_conflict() -> None:
    with pytest.raises(EvidenceConflict, match=f"allocation {LEASE}: Slurm lists job 43"):
        observe_step(STEP, LEASE, JOB, [QueueRow(43, "RUNNING")], (), ())


# --- Property --------------------------------------------------------------------------------

MOMENTS = st.integers(0, 40).map(lambda minute: AT + at(minute))
JOBS = st.sampled_from((JOB, JOB, JOB, 43))
OWN = st.one_of(
    st.builds(Start, JOBS, MOMENTS),
    st.builds(Stop, JOBS, MOMENTS),
    st.builds(Void, JOBS, MOMENTS),
    st.builds(End, JOBS, st.sampled_from((0, 0, 1)), st.sampled_from((0, 1, 75, 143)), MOMENTS),
    st.builds(Limit, JOBS, MOMENTS),
    st.builds(Handoff, st.just(0), MOMENTS),
)
LEASE_MARKERS = st.one_of(
    st.builds(Start, st.just(JOB), MOMENTS),
    st.builds(Stop, st.just(JOB), MOMENTS),
    st.builds(Limit, st.just(JOB), MOMENTS),
)
STATES = st.sampled_from((None, None, "PENDING", "RUNNING", "COMPLETING", "COMPLETED", "TIMEOUT"))


@settings(max_examples=1000)
@given(
    STATES,
    st.lists(OWN, max_size=4),
    st.lists(LEASE_MARKERS, max_size=3),
    MOMENTS,
    st.integers(1, 40).map(at),
)
def test_step_evidence_is_a_conflict_or_a_consistent_execution(
    state: str | None,
    own: list[Marker],
    lease: list[Marker],
    now: datetime,
    bound: timedelta,
) -> None:
    seen = None
    with contextlib.suppress(EvidenceConflict):
        seen = step(state, own, lease)
    if seen is None:
        return
    execution, _, _ = seen.execution(0)
    if seen.live:
        assert not execution.terminal
    spent, _ = seen.run_time(0, now, bound)
    assert timedelta(0) <= spent <= bound
