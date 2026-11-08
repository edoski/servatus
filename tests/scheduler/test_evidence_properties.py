"""Property tests: evidence parsing fails closed, and execution follows the spec's table for every
combination of queue row and run markers."""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath

from hypothesis import given, settings
from hypothesis import strategies as st
from scheduler_fixtures import ALLOCATION, IDENTITY, LOG_ROOT, allocation

from servatus.campaign._evidence import (
    End,
    Execution,
    Handoff,
    Limit,
    Marker,
    Observation,
    Start,
    StepEnd,
    Stop,
    StopReason,
    observe,
    parse_markers,
    parse_queue,
    parse_receipt,
)
from servatus.errors import EvidenceConflict

ROOT = PurePosixPath(LOG_ROOT)
RUNNING = ("RUNNING", "COMPLETING", "SIGNALING", "STAGE_OUT", "SUSPENDED", "STOPPED")
SETTLING = (
    "COMPLETED",
    "CANCELLED",
    "FAILED",
    "TIMEOUT",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "OUT_OF_MEMORY",
)
KNOWN = frozenset(RUNNING + SETTLING)
STATES = (
    st.none()
    | st.sampled_from(RUNNING + SETTLING + ("PENDING", "REQUEUE_HOLD"))
    | (st.from_regex(r"[A-Z][A-Z_]{0,15}", fullmatch=True))
)
CODES = st.none() | st.just(75) | st.integers(0, 255)
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
MOMENTS = st.integers(0, 4).map(lambda minute: EPOCH + timedelta(minutes=minute))
WHEN = st.none() | MOMENTS


@st.composite
def observations(draw: st.DrawFn) -> Observation:
    codes = draw(st.lists(CODES, min_size=1, max_size=6))
    return Observation(
        draw(STATES),
        draw(WHEN),
        draw(WHEN),
        draw(WHEN),
        tuple(None if code is None else StepEnd(code, draw(MOMENTS)) for code in codes),
        tuple(draw(st.booleans()) for _ in codes),
    )


def expected(seen: Observation, slot: int) -> tuple[Execution, int | None, StopReason | None]:
    """The spec's table, row by row."""
    state, end = seen.slurm_state, seen.ends[slot]
    if state is not None and state not in KNOWN:
        return Execution.QUEUED, None, None
    if not seen.handed[slot]:
        return Execution.STOPPED, None, StopReason.NEVER_STARTED
    if end is not None:
        code = end.exit_code
        if code == 0:
            return Execution.SUCCEEDED, 0, None
        if code == 75:
            return Execution.STOPPED, 75, StopReason.YIELDED
        if seen.limit_at is not None and seen.limit_at <= end.at:
            return Execution.STOPPED, code, StopReason.TIME_LIMIT
        return Execution.FAILED, code, None
    if state is not None:
        return (Execution.ENDING if state in SETTLING else Execution.RUNNING), None, None
    if seen.started_at is None:
        return Execution.STOPPED, None, StopReason.NEVER_STARTED
    if seen.stopped_at is None:
        return Execution.LOST, None, None
    return Execution.STOPPED, None, None if seen.limit_at is None else StopReason.TIME_LIMIT


@settings(max_examples=2000)
@given(observations(), st.data())
def test_execution_follows_the_spec_table(seen: Observation, data: st.DataObject) -> None:
    slot = data.draw(st.integers(0, len(seen.ends) - 1))
    execution, code, reason = seen.execution(slot)
    assert (execution, code, reason) == expected(seen, slot)
    assert code is None or execution in (Execution.SUCCEEDED, Execution.FAILED, Execution.STOPPED)
    assert reason is None or execution is Execution.STOPPED
    if seen.live and seen.handed[slot]:
        assert execution not in (Execution.STOPPED, Execution.LOST) or code is not None
    if not seen.live:
        assert execution not in (Execution.QUEUED, Execution.RUNNING, Execution.ENDING)


@settings(max_examples=1000)
@given(observations(), st.data())
def test_run_time_stays_within_the_bound_and_is_exact_only_when_finished(
    seen: Observation, data: st.DataObject
) -> None:
    slot = data.draw(st.integers(0, len(seen.ends) - 1))
    bound = timedelta(minutes=data.draw(st.integers(1, 5)))
    now = data.draw(MOMENTS)
    spent, exact = seen.run_time(slot, now, bound)
    assert timedelta(0) <= spent <= bound
    if not exact:
        assert seen.handed[slot] and seen.started_at is not None
        assert seen.ends[slot] is None and seen.stopped_at is None
        assert seen.live or spent == bound


@settings(max_examples=500)
@given(observations(), st.data())
def test_a_slot_is_decided_by_its_own_markers_alone(seen: Observation, data: st.DataObject) -> None:
    slot = data.draw(st.integers(0, len(seen.ends) - 1))
    count = len(seen.ends)
    others = list(data.draw(observations()).ends[:count])
    others += [None] * (count - len(others))
    others[slot] = seen.ends[slot]
    handed = [data.draw(st.booleans()) for _ in range(count)]
    handed[slot] = seen.handed[slot]
    changed = Observation(
        seen.slurm_state,
        seen.started_at,
        seen.stopped_at,
        seen.limit_at,
        tuple(others),
        tuple(handed),
    )
    assert changed.execution(slot) == seen.execution(slot)


def marker_name(allocation_id: str, marker: Marker) -> str:
    if isinstance(marker, Handoff):
        return f"{allocation_id}.{marker.slot}.go"
    if isinstance(marker, End):
        return f"{allocation_id}.{marker.job}.{marker.slot}.{marker.exit_code}.end"
    kind = {Start: "start", Stop: "stop", Limit: "limit"}[type(marker)]
    return f"{allocation_id}.{marker.job}.{kind}"


def mtime(at: datetime) -> str:
    elapsed = at - EPOCH
    return f"{elapsed // timedelta(seconds=1)}.{elapsed.microseconds:06d}"


JOBS = st.integers(1, 10**12)
STAMPS = st.integers(0, 10**15).map(lambda micro: EPOCH + timedelta(microseconds=micro))
MARKERS: st.SearchStrategy[Marker] = st.one_of(
    st.builds(Start, JOBS, STAMPS),
    st.builds(Stop, JOBS, STAMPS),
    st.builds(Limit, JOBS, STAMPS),
    st.builds(End, JOBS, st.integers(0, 10**6), st.integers(0, 255), STAMPS),
    st.builds(Handoff, st.integers(0, 10**6), STAMPS),
)
IDS = [allocation(index) for index in range(1, 5)]


@settings(max_examples=500)
@given(
    st.lists(st.tuples(st.sampled_from(IDS), MARKERS), max_size=12),
    st.sets(st.sampled_from(IDS), min_size=1),
    st.sampled_from([LOG_ROOT, "/", "/a b/c"]),
)
def test_markers_round_trip_and_only_queried_allocations_are_returned(
    written: list[tuple[str, Marker]], queried: set[str], root: str
) -> None:
    prefix = root.rstrip("/")
    noise = [f"7 /elsewhere/{ALLOCATION}.1.start", f"7.5 {prefix}/{ALLOCATION}-1.out"]
    lines = [
        f"{mtime(marker.at)} {prefix}/{marker_name(owner, marker)}" for owner, marker in written
    ] + noise
    output = "".join(f"{line}\n" for line in lines).encode()
    found = parse_markers(output, PurePosixPath(root), queried)
    assert found == {
        allocation_id: [marker for owner, marker in written if owner == allocation_id]
        for allocation_id in queried
    }


@settings(max_examples=500)
@given(
    st.lists(
        st.tuples(st.sampled_from(["start", "stop", "limit", "end", "go"]), st.integers(0, 3)),
        max_size=8,
    ),
    st.sampled_from([None, "RUNNING", "COMPLETED", "PENDING"]),
    st.booleans(),
)
def test_observing_markers_of_one_job_never_guesses(
    events: list[tuple[str, int]], state: str | None, gated: bool
) -> None:
    markers: list[Marker] = []
    for step, (kind, slot) in enumerate(events):
        at = EPOCH + timedelta(minutes=step)
        markers.append(
            {
                "start": Start(42, at),
                "stop": Stop(42, at),
                "limit": Limit(42, at),
                "end": End(42, slot, slot, at),
                "go": Handoff(slot, at),
            }[kind]
        )
    try:
        seen = observe(ALLOCATION, 42, 3, [], markers, gated=gated)
    except EvidenceConflict as error:
        ends = [marker.slot for marker in markers if isinstance(marker, End)]
        handoffs = [marker.slot for marker in markers if isinstance(marker, Handoff)]
        assert (
            max(ends + [0]) >= 3
            or len(set(ends)) < len(ends)
            or (handoffs and (not gated or max(handoffs) >= 3))
        ), error
        return
    assert (seen.started_at is not None) == any(isinstance(marker, Start) for marker in markers)
    assert (seen.stopped_at is not None) == any(isinstance(marker, Stop) for marker in markers)
    for slot in range(3):
        execution, code, _ = seen.execution(slot)
        handoffs = [m.at for m in markers if isinstance(m, Handoff) and m.slot == slot]
        handed = not gated or bool(
            handoffs and (seen.started_at is None or handoffs[-1] <= seen.started_at)
        )
        assert seen.handed[slot] == handed
        ended = any(isinstance(marker, End) and marker.slot == slot for marker in markers)
        assert code is None or ended
        if execution is Execution.LOST:
            assert seen.started_at is not None and seen.stopped_at is None


@settings(max_examples=1000)
@given(st.binary(max_size=256))
def test_parsers_raise_only_evidence_conflict_on_arbitrary_bytes(data: bytes) -> None:
    with contextlib.suppress(EvidenceConflict):
        assert parse_receipt(data) > 0
    with contextlib.suppress(EvidenceConflict):
        parse_queue(data, [IDENTITY])
    with contextlib.suppress(EvidenceConflict):
        parse_markers(data, ROOT, [ALLOCATION])


FIELDS = st.sampled_from(
    [
        "42",
        "0",
        "042",
        "-1",
        " 42",
        "",
        IDENTITY,
        f"{IDENTITY}-0",
        "servatus-x",
        "RUNNING",
        "PENDING",
        "CANCELLED by 1",
        "é",
        "a\tb",
    ]
)


@settings(max_examples=2000)
@given(st.lists(st.lists(FIELDS, min_size=2, max_size=4).map("|".join), max_size=4))
def test_queue_rows_are_trusted_only_for_the_queried_identity(rows: list[str]) -> None:
    output = "".join(f"{row}\n" for row in rows).encode()
    try:
        parsed = parse_queue(output, [IDENTITY])
    except EvidenceConflict:
        return
    assert len(parsed[IDENTITY]) == len(rows)
    for row in parsed[IDENTITY]:
        assert f"{row.job}|{IDENTITY}|{row.state}\n".encode() in output
