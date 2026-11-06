"""Property tests: evidence parsing fails closed, and execution follows the spec's table for every
combination of queue row and run markers."""

from __future__ import annotations

import contextlib
from pathlib import PurePosixPath

from hypothesis import given, settings
from hypothesis import strategies as st
from scheduler_fixtures import ALLOCATION, IDENTITY, LOG_ROOT, allocation

from servatus.campaign._evidence import (
    End,
    Execution,
    Marker,
    Observation,
    Start,
    Stop,
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
CODES = st.none() | st.integers(0, 255)


@st.composite
def observations(draw: st.DrawFn) -> Observation:
    codes = draw(st.lists(CODES, min_size=1, max_size=6))
    return Observation(draw(STATES), draw(st.booleans()), draw(st.booleans()), tuple(codes))


def expected(seen: Observation, slot: int) -> tuple[Execution, int | None]:
    """The spec's table, row by row."""
    state, code = seen.slurm_state, seen.exit_codes[slot]
    if state is not None and state not in KNOWN:
        return Execution.QUEUED, None
    if code is not None:
        return (Execution.SUCCEEDED if code == 0 else Execution.FAILED), code
    if state is not None:
        return (Execution.ENDING if state in SETTLING else Execution.RUNNING), None
    if not seen.started:
        return Execution.STOPPED, None
    return (Execution.STOPPED if seen.stopped else Execution.LOST), None


@settings(max_examples=2000)
@given(observations(), st.data())
def test_execution_follows_the_spec_table(seen: Observation, data: st.DataObject) -> None:
    slot = data.draw(st.integers(0, len(seen.exit_codes) - 1))
    execution, code = seen.execution(slot)
    assert (execution, code) == expected(seen, slot)
    assert (code is not None) == (execution in (Execution.SUCCEEDED, Execution.FAILED))
    if seen.live:
        assert execution not in (Execution.STOPPED, Execution.LOST)
    else:
        assert execution not in (Execution.QUEUED, Execution.RUNNING, Execution.ENDING)


@settings(max_examples=500)
@given(observations(), st.data())
def test_a_slot_is_decided_by_its_own_end_marker_alone(
    seen: Observation, data: st.DataObject
) -> None:
    slot = data.draw(st.integers(0, len(seen.exit_codes) - 1))
    others = data.draw(st.lists(CODES, min_size=len(seen.exit_codes), max_size=6))
    others[slot] = seen.exit_codes[slot]
    changed = Observation(seen.slurm_state, seen.started, seen.stopped, tuple(others))
    assert changed.execution(slot) == seen.execution(slot)


def marker_name(allocation_id: str, marker: Marker) -> str:
    if isinstance(marker, Start):
        return f"{allocation_id}.{marker.job}.start"
    if isinstance(marker, Stop):
        return f"{allocation_id}.{marker.job}.stop"
    return f"{allocation_id}.{marker.job}.{marker.slot}.{marker.exit_code}.end"


JOBS = st.integers(1, 10**12)
MARKERS: st.SearchStrategy[Marker] = st.one_of(
    st.builds(Start, JOBS),
    st.builds(Stop, JOBS),
    st.builds(End, JOBS, st.integers(0, 10**6), st.integers(0, 255)),
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
    noise = [f"/elsewhere/{ALLOCATION}.1.start", f"{prefix}/{ALLOCATION}-1.out"]
    lines = [f"{prefix}/{marker_name(owner, marker)}" for owner, marker in written] + noise
    output = "".join(f"{line}\n" for line in lines).encode()
    found = parse_markers(output, PurePosixPath(root), queried)
    assert found == {
        allocation_id: [marker for owner, marker in written if owner == allocation_id]
        for allocation_id in queried
    }


@settings(max_examples=500)
@given(
    st.lists(st.tuples(st.sampled_from(["start", "stop", "end"]), st.integers(0, 3)), max_size=8),
    st.sampled_from([None, "RUNNING", "COMPLETED", "PENDING"]),
)
def test_observing_markers_of_one_job_never_guesses(
    events: list[tuple[str, int]], state: str | None
) -> None:
    markers: list[Marker] = []
    for kind, slot in events:
        markers.append(
            Start(42) if kind == "start" else Stop(42) if kind == "stop" else End(42, slot, slot)
        )
    try:
        seen = observe(ALLOCATION, 42, 3, [], markers)
    except EvidenceConflict as error:
        ends = [marker.slot for marker in markers if isinstance(marker, End)]
        assert max(ends) >= 3 or len(set(ends)) < len(ends), error
        return
    assert seen.started == any(isinstance(marker, Start) for marker in markers)
    assert seen.stopped == any(isinstance(marker, Stop) for marker in markers)
    for slot in range(3):
        execution, code = seen.execution(slot)
        assert (code is not None) == any(
            isinstance(marker, End) and marker.slot == slot for marker in markers
        )
        if execution is Execution.LOST:
            assert seen.started and not seen.stopped


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
