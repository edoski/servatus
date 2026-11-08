from __future__ import annotations

from datetime import timedelta
from pathlib import PurePosixPath

import pytest
from scheduler_fixtures import ALLOCATION, AT, IDENTITY, LOG_ROOT, MTIME, allocation, listing

from servatus.campaign._evidence import (
    MAX_REPLY_LINES,
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
    evidenced_job,
    observe,
    parse_markers,
    parse_queue,
    parse_receipt,
    queue_row,
)
from servatus.errors import EvidenceConflict

ROOT = PurePosixPath(LOG_ROOT)
OTHER = allocation(5)

# The spec's Slurm state families, written out independently of the implementation's sets.
QUEUED_STATES = (
    "PENDING",
    "CONFIGURING",
    "REQUEUED",
    "REQUEUE_HOLD",
    "RESV_DEL_HOLD",
    "SPECIAL_EXIT",
    "RESIZING",
    "FUTURE_KEYWORD",
)
RUNNING_STATES = ("RUNNING", "COMPLETING", "SIGNALING", "STAGE_OUT", "SUSPENDED", "STOPPED")
SETTLING_STATES = (
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


# --- sbatch receipt --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "expected"),
    [(b"42", 42), (b"42\n", 42), (b"9876543210\n", 9876543210), (b"1\n", 1)],
)
def test_receipt_is_one_job_number_with_an_optional_newline(output: bytes, expected: int) -> None:
    assert parse_receipt(output) == expected


@pytest.mark.parametrize(
    "output",
    [
        b"",
        b"\n",
        b"42;cluster\n",
        b"42;cluster",
        b"042\n",
        b"0\n",
        b"-1\n",
        b" 42\n",
        b"42 \n",
        b"42\n\n",
        b"42\r\n",
        b"\n42",
        b"4\n2\n",
        b"Submitted batch job 42\n",
        b"\xd9\xa4\xd9\xa2\n",
    ],
)
def test_receipt_rejects_anything_but_digits(output: bytes) -> None:
    with pytest.raises(EvidenceConflict, match="exactly one job number"):
        parse_receipt(output)


# --- squeue rows -----------------------------------------------------------------------------


def test_queue_rows_are_grouped_by_queried_name() -> None:
    other = f"servatus-{OTHER}"
    output = f"42|{IDENTITY}|RUNNING\n43|{IDENTITY}|PENDING\n7|{other}|COMPLETED\n".encode()
    rows = parse_queue(output, [IDENTITY, other, "servatus-unseen"])
    assert rows == {
        IDENTITY: [QueueRow(42, "RUNNING"), QueueRow(43, "PENDING")],
        other: [QueueRow(7, "COMPLETED")],
        "servatus-unseen": [],
    }


def test_empty_queue_reply_lists_every_name_without_rows() -> None:
    assert parse_queue(b"", [IDENTITY]) == {IDENTITY: []}


def test_queue_row_for_an_unqueried_name_is_a_conflict() -> None:
    with pytest.raises(EvidenceConflict, match="row for an unrelated job"):
        parse_queue(b"42|servatus-other|RUNNING\n", [IDENTITY])


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (f"42|{IDENTITY}", "row is malformed"),
        (f"42|{IDENTITY}|RUNNING|extra", "row is malformed"),
        (f"042|{IDENTITY}|RUNNING", "row is malformed"),
        (f"0|{IDENTITY}|RUNNING", "row is malformed"),
        (f"42.0|{IDENTITY}|RUNNING", "row is malformed"),
        (f"|{IDENTITY}|RUNNING", "row is malformed"),
        (f" 42|{IDENTITY}|RUNNING", "row is malformed"),
        (f"42|{IDENTITY}\t|RUNNING", "row is malformed"),
        (f"42|{IDENTITY}|RUN\x7fNING", "row is malformed"),
        (f"42|{IDENTITY}|", "state is malformed"),
        (f"42|{IDENTITY}|CANCELLED by 7", "state is malformed"),
        (f"42|{IDENTITY}|RUNNING+", "state is malformed"),
        (f"42|{IDENTITY}|-RUNNING", "state is malformed"),
    ],
)
def test_malformed_queue_rows_are_conflicts(row: str, message: str) -> None:
    with pytest.raises(EvidenceConflict, match=message):
        parse_queue(f"{row}\n".encode(), [IDENTITY])


def test_queue_row_that_is_not_utf8_is_a_conflict() -> None:
    with pytest.raises(EvidenceConflict, match="row is malformed"):
        parse_queue(b"42|servatus-\xff|RUNNING\n", [IDENTITY])


def test_partial_queue_reply_is_a_conflict() -> None:
    with pytest.raises(EvidenceConflict, match="output is partial"):
        parse_queue(f"42|{IDENTITY}|RUNNING".encode(), [IDENTITY])


def test_queue_reply_line_bound_is_enforced() -> None:
    row = f"42|{IDENTITY}|RUNNING\n".encode()
    assert len(parse_queue(row * MAX_REPLY_LINES, [IDENTITY])[IDENTITY]) == MAX_REPLY_LINES
    with pytest.raises(EvidenceConflict, match="exceeds its line bound"):
        parse_queue(row * (MAX_REPLY_LINES + 1), [IDENTITY])


@pytest.mark.parametrize("state", SETTLING_STATES)
def test_terminal_rows_are_settling(state: str) -> None:
    assert QueueRow(42, state).settling


@pytest.mark.parametrize("state", QUEUED_STATES + RUNNING_STATES)
def test_other_rows_are_not_settling(state: str) -> None:
    assert not QueueRow(42, state).settling


# --- Run markers -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "marker"),
    [
        (f"{ALLOCATION}.42.start", Start(42, AT)),
        (f"{ALLOCATION}.42.stop", Stop(42, AT)),
        (f"{ALLOCATION}.42.0.0.end", End(42, 0, 0, AT)),
        (f"{ALLOCATION}.42.17.3.end", End(42, 17, 3, AT)),
        (f"{ALLOCATION}.9876543210.3.255.end", End(9876543210, 3, 255, AT)),
    ],
)
def test_each_marker_kind_parses(name: str, marker: Marker) -> None:
    assert parse_markers(listing(name), ROOT, [ALLOCATION]) == {ALLOCATION: [marker]}


@pytest.mark.parametrize("code", range(256))
def test_every_exit_status_parses(code: int) -> None:
    markers = parse_markers(listing(f"{ALLOCATION}.42.1.{code}.end"), ROOT, [ALLOCATION])
    assert markers == {ALLOCATION: [End(42, 1, code, AT)]}


@pytest.mark.parametrize(
    "suffix",
    [
        "42.0.256.end",
        "42.0.999.end",
        "42.0.-1.end",
        "42.0.01.end",
        "42.0.00.end",
        "42.01.0.end",
        "042.start",
        "042.0.0.end",
        "0.start",
        "start",
        "42.end",
        "42.0.end",
        "42.x.0.end",
        "42.0.0.0.end",
        "42.start.tmp",
        "42.Start",
        "42.stop.end",
        "４２.start",
        "42.0.0.end\x00",
        "42.limit.end",
        "01.go",
        "42.0.go",
        "go",
        "x.go",
    ],
)
def test_malformed_marker_of_a_queried_allocation_is_a_conflict(suffix: str) -> None:
    with pytest.raises(EvidenceConflict, match=f"allocation {ALLOCATION} has a malformed"):
        parse_markers(listing(f"{ALLOCATION}.{suffix}"), ROOT, [ALLOCATION])


def test_marker_name_that_is_not_ascii_is_a_conflict() -> None:
    output = f"{MTIME} {LOG_ROOT}/{ALLOCATION}.42.".encode() + b"\xff.start\n"
    with pytest.raises(EvidenceConflict, match=f"allocation {ALLOCATION} has a malformed"):
        parse_markers(output, ROOT, [ALLOCATION])


def test_markers_of_other_allocations_and_unrelated_files_are_ignored() -> None:
    output = listing(
        f"{OTHER}.42.start",
        f"{OTHER}.junk.end",
        f"{ALLOCATION}-42.out",
        f"{ALLOCATION}-42-0.out",
        f"x{ALLOCATION}.42.start",
        f"{ALLOCATION.upper()}.42.start",
        f"{ALLOCATION}0.42.start",
        "README.start",
        f"{ALLOCATION}.42.start",
    )
    assert parse_markers(output, ROOT, [ALLOCATION]) == {ALLOCATION: [Start(42, AT)]}


def test_lines_outside_the_log_root_are_ignored() -> None:
    output = (
        listing(f"{ALLOCATION}.43.start", root="/elsewhere")
        + listing(f"{ALLOCATION}.43.start", root=f"{LOG_ROOT}2")
        + listing(f"sub/{ALLOCATION}.43.start")
        + listing(f"{ALLOCATION}.43.start", root=LOG_ROOT.removeprefix("/"))
        + listing(f"{ALLOCATION}.42.stop")
    )
    assert parse_markers(output, ROOT, [ALLOCATION]) == {ALLOCATION: [Stop(42, AT)]}


@pytest.mark.parametrize("root", [LOG_ROOT, f"{LOG_ROOT}/", "/"])
def test_log_root_matches_with_or_without_a_trailing_slash(root: str) -> None:
    shown = root.rstrip("/")
    output = f"{MTIME} {shown}/{ALLOCATION}.42.start\n".encode()
    assert parse_markers(output, PurePosixPath(root), [ALLOCATION]) == {ALLOCATION: [Start(42, AT)]}


def test_markers_are_grouped_by_allocation_in_listing_order() -> None:
    output = listing(
        f"{ALLOCATION}.42.1.0.end",
        f"{OTHER}.7.start",
        f"{ALLOCATION}.42.start",
        f"{ALLOCATION}.42.0.9.end",
    )
    assert parse_markers(output, ROOT, [ALLOCATION, OTHER, allocation(6)]) == {
        ALLOCATION: [End(42, 1, 0, AT), Start(42, AT), End(42, 0, 9, AT)],
        OTHER: [Start(7, AT)],
        allocation(6): [],
    }


def test_markers_carry_their_mtime_and_every_kind_parses() -> None:
    output = listing(
        f"{ALLOCATION}.42.limit", f"{ALLOCATION}.3.go", mtime="1893456061.5"
    ) + listing(f"{ALLOCATION}.42.start", mtime="1893456000.1234567891")
    assert parse_markers(output, ROOT, [ALLOCATION]) == {
        ALLOCATION: [
            Limit(42, AT + timedelta(seconds=61.5)),
            Handoff(3, AT + timedelta(seconds=61.5)),
            Start(42, AT + timedelta(microseconds=123456)),
        ]
    }


@pytest.mark.parametrize(
    "line",
    [
        f"{LOG_ROOT}/{ALLOCATION}.42.start",
        f"1893456000.x {LOG_ROOT}/{ALLOCATION}.42.start",
        f"-1 {LOG_ROOT}/{ALLOCATION}.42.start",
        "1893456000 ",
        "1893456000",
        f"9999999999999 {LOG_ROOT}/{ALLOCATION}.42.start",
    ],
)
def test_a_listing_line_without_an_mtime_is_a_conflict(line: str) -> None:
    with pytest.raises(EvidenceConflict, match="run marker listing is malformed"):
        parse_markers(f"{line}\n".encode(), ROOT, [ALLOCATION])


def test_partial_marker_listing_is_a_conflict() -> None:
    with pytest.raises(EvidenceConflict, match="output is partial"):
        parse_markers(listing(f"{ALLOCATION}.42.start").rstrip(b"\n"), ROOT, [ALLOCATION])


# --- Observation -----------------------------------------------------------------------------


def observation(
    state: str | None,
    *,
    started: bool,
    stopped: bool,
    code: int | None,
    limit: timedelta | None = None,
    handed: bool = True,
) -> Observation:
    """One slot that started at ``AT``, ended 10 minutes later with ``code``, stopped 20 minutes
    later, and got the pre-limit signal ``limit`` after ``AT``."""
    return Observation(
        state,
        AT if started else None,
        AT + timedelta(minutes=20) if stopped else None,
        None if limit is None else AT + limit,
        (None if code is None else StepEnd(code, AT + timedelta(minutes=10)),),
        (handed,),
    )


MARKERS = [
    pytest.param(started, stopped, code, id=f"start={started}-stop={stopped}-end={code}")
    for started in (False, True)
    for stopped in (False, True)
    for code in (None, 0, 1, 255)
]


@pytest.mark.parametrize("state", QUEUED_STATES)
@pytest.mark.parametrize(("started", "stopped", "code"), MARKERS)
def test_a_queued_row_is_queued_whatever_the_markers(
    state: str, started: bool, stopped: bool, code: int | None
) -> None:
    seen = observation(state, started=started, stopped=stopped, code=code)
    assert seen.execution(0) == (Execution.QUEUED, None, None)
    assert seen.live


@pytest.mark.parametrize("state", RUNNING_STATES + SETTLING_STATES)
@pytest.mark.parametrize(("started", "stopped"), [(False, False), (True, False), (True, True)])
def test_a_running_or_settling_row_follows_the_slot_end_marker(
    state: str, started: bool, stopped: bool
) -> None:
    def execution(code: int | None) -> tuple[Execution, int | None, StopReason | None]:
        return observation(state, started=started, stopped=stopped, code=code).execution(0)

    ending = state in SETTLING_STATES
    assert execution(None) == (Execution.ENDING if ending else Execution.RUNNING, None, None)
    assert execution(0) == (Execution.SUCCEEDED, 0, None)
    assert execution(1) == (Execution.FAILED, 1, None)
    assert execution(255) == (Execution.FAILED, 255, None)
    assert execution(75) == (Execution.STOPPED, 75, StopReason.YIELDED)
    assert observation(state, started=started, stopped=stopped, code=None).live


@pytest.mark.parametrize(("started", "stopped"), [(False, False), (True, False), (True, True)])
def test_a_purged_job_with_a_slot_end_marker_has_that_outcome(started: bool, stopped: bool) -> None:
    def execution(code: int) -> tuple[Execution, int | None, StopReason | None]:
        return observation(None, started=started, stopped=stopped, code=code).execution(0)

    assert execution(0) == (Execution.SUCCEEDED, 0, None)
    assert execution(7) == (Execution.FAILED, 7, None)
    assert execution(75) == (Execution.STOPPED, 75, StopReason.YIELDED)


@pytest.mark.parametrize(
    ("limit", "code", "expected"),
    [
        (timedelta(minutes=5), 138, (Execution.STOPPED, 138, StopReason.TIME_LIMIT)),
        (timedelta(minutes=10), 1, (Execution.STOPPED, 1, StopReason.TIME_LIMIT)),
        (timedelta(minutes=11), 1, (Execution.FAILED, 1, None)),
        (timedelta(minutes=5), 0, (Execution.SUCCEEDED, 0, None)),
        (timedelta(minutes=5), 75, (Execution.STOPPED, 75, StopReason.YIELDED)),
        (None, 138, (Execution.FAILED, 138, None)),
    ],
    ids=["after-signal", "same-instant", "before-signal", "done", "yielded", "no-signal"],
)
def test_a_nonzero_exit_after_the_pre_limit_signal_is_a_time_limit_stop(
    limit: timedelta | None, code: int, expected: tuple[Execution, int, StopReason | None]
) -> None:
    seen = observation(None, started=True, stopped=False, code=code, limit=limit)
    assert seen.execution(0) == expected


@pytest.mark.parametrize(
    ("started", "stopped", "limit", "expected"),
    [
        (True, True, None, (Execution.STOPPED, None, None)),
        (True, True, timedelta(minutes=5), (Execution.STOPPED, None, StopReason.TIME_LIMIT)),
        (False, False, None, (Execution.STOPPED, None, StopReason.NEVER_STARTED)),
        (False, True, None, (Execution.STOPPED, None, StopReason.NEVER_STARTED)),
        (True, False, None, (Execution.LOST, None, None)),
        (True, False, timedelta(minutes=5), (Execution.LOST, None, None)),
    ],
    ids=["interrupted", "time-limit", "never-started", "stopped-before-start", "lost", "lost-late"],
)
def test_a_purged_job_without_a_slot_end_marker(
    started: bool,
    stopped: bool,
    limit: timedelta | None,
    expected: tuple[Execution, None, StopReason | None],
) -> None:
    seen = observation(None, started=started, stopped=stopped, code=None, limit=limit)
    assert seen.execution(0) == expected
    assert not seen.live


@pytest.mark.parametrize("state", [None, "RUNNING", "COMPLETED"])
def test_a_slot_never_handed_to_a_link_never_started_there(state: str | None) -> None:
    seen = observation(state, started=True, stopped=False, code=None, handed=False)
    assert seen.execution(0) == (Execution.STOPPED, None, StopReason.NEVER_STARTED)
    assert seen.run_time(0, AT + timedelta(hours=1), timedelta(hours=2)) == (timedelta(0), True)


def test_each_slot_is_decided_by_its_own_end_marker() -> None:
    ends = (StepEnd(0, AT), None, StepEnd(3, AT))
    seen = Observation(None, AT, None, None, ends, (True, True, True))
    assert [seen.execution(slot) for slot in range(3)] == [
        (Execution.SUCCEEDED, 0, None),
        (Execution.LOST, None, None),
        (Execution.FAILED, 3, None),
    ]


BOUND = timedelta(minutes=30)


@pytest.mark.parametrize(
    ("state", "started", "stopped", "code", "now", "expected"),
    [
        (None, True, False, 0, 60, (timedelta(minutes=10), True)),
        (None, True, True, None, 60, (timedelta(minutes=20), True)),
        ("COMPLETED", True, True, 0, 60, (timedelta(minutes=10), True)),
        ("RUNNING", True, False, None, 7, (timedelta(minutes=7), False)),
        ("RUNNING", True, False, None, 90, (BOUND, False)),
        ("RUNNING", True, False, None, -5, (timedelta(0), False)),
        ("PENDING", False, False, None, 60, (timedelta(0), True)),
        (None, False, False, None, 60, (timedelta(0), True)),
        (None, True, False, None, 60, (BOUND, False)),
    ],
    ids=["ended", "stopped", "settling", "running", "clamped", "skewed", "queued", "never", "lost"],
)
def test_run_time_is_the_span_between_marker_mtimes(
    state: str | None,
    started: bool,
    stopped: bool,
    code: int | None,
    now: int,
    expected: tuple[timedelta, bool],
) -> None:
    seen = observation(state, started=started, stopped=stopped, code=code)
    assert seen.run_time(0, AT + timedelta(minutes=now), BOUND) == expected


def test_only_known_outcomes_are_terminal() -> None:
    assert {state for state in Execution if state.terminal} == {
        Execution.SUCCEEDED,
        Execution.FAILED,
        Execution.STOPPED,
    }


# --- Decisions -------------------------------------------------------------------------------


def test_queue_row_is_the_single_row_of_the_recorded_job() -> None:
    row = QueueRow(42, "RUNNING")
    assert queue_row(ALLOCATION, [], 42) is None
    assert queue_row(ALLOCATION, [], None) is None
    assert queue_row(ALLOCATION, [row], 42) == row
    assert queue_row(ALLOCATION, [row], None) == row


@pytest.mark.parametrize(
    ("rows", "job", "message"),
    [
        ([QueueRow(43, "RUNNING")], 42, "lists job 43 under its identity, not the accepted job 42"),
        ([QueueRow(42, "RUNNING"), QueueRow(42, "PENDING")], 42, "lists job 42 twice"),
        ([QueueRow(42, "RUNNING"), QueueRow(43, "PENDING")], 42, "lists jobs 42, 43"),
        ([QueueRow(44, "RUNNING"), QueueRow(43, "PENDING")], None, "lists jobs 43, 44"),
    ],
)
def test_queue_row_conflicts_name_the_allocation(
    rows: list[QueueRow], job: int | None, message: str
) -> None:
    with pytest.raises(EvidenceConflict, match=f"allocation {ALLOCATION}: Slurm {message}"):
        queue_row(ALLOCATION, rows, job)


def test_observe_combines_the_row_and_the_markers() -> None:
    later = AT + timedelta(minutes=1)
    markers = [Start(42, AT), End(42, 2, 0, later), End(42, 0, 5, AT), Limit(42, later)]
    seen = observe(ALLOCATION, 42, 3, [QueueRow(42, "COMPLETING")], markers)
    ends = (StepEnd(5, AT), None, StepEnd(0, later))
    assert seen == Observation("COMPLETING", AT, None, later, ends, (True, True, True))
    assert observe(ALLOCATION, 42, 2, [], [Start(42, AT), Stop(42, later)]) == Observation(
        None, AT, later, None, (None, None), (True, True)
    )
    assert observe(ALLOCATION, 42, 1, [], []) == Observation(
        None, None, None, None, (None,), (True,)
    )


def test_a_gated_link_has_exactly_the_slots_handed_to_it_before_it_started() -> None:
    before, after = AT - timedelta(seconds=1), AT + timedelta(seconds=1)
    markers = [Handoff(0, before), Handoff(1, AT), Handoff(3, after), Start(42, AT)]
    seen = observe(ALLOCATION, 42, 4, [], markers, gated=True)
    assert seen.handed == (True, True, False, False)
    pending = observe(ALLOCATION, 42, 2, [QueueRow(42, "PENDING")], [Handoff(1, AT)], gated=True)
    assert pending.handed == (False, True)


@pytest.mark.parametrize(
    ("rows", "markers", "message"),
    [
        ([QueueRow(43, "RUNNING")], [], "Slurm lists job 43 under its identity"),
        ([QueueRow(42, "RUNNING"), QueueRow(42, "RUNNING")], [], "lists job 42 twice"),
        ([], [Start(43, AT)], "a run marker names job 43, not the accepted job 42"),
        ([], [Stop(43, AT)], "a run marker names job 43"),
        ([], [End(43, 0, 0, AT)], "a run marker names job 43"),
        ([], [End(42, 2, 0, AT)], "unexpected end marker for slot 2"),
        ([], [End(42, 5, 0, AT)], "unexpected end marker for slot 5"),
        ([], [End(42, 1, 0, AT), End(42, 1, 1, AT)], "unexpected end marker for slot 1"),
        ([], [Limit(43, AT)], "a run marker names job 43"),
        ([], [Handoff(0, AT)], "unexpected handoff marker for slot 0"),
    ],
)
def test_observe_conflicts_name_the_allocation(
    rows: list[QueueRow], markers: list[Marker], message: str
) -> None:
    with pytest.raises(EvidenceConflict, match=f"allocation {ALLOCATION}: .*{message}"):
        observe(ALLOCATION, 42, 2, rows, markers)


def test_a_handoff_to_a_slot_a_gated_link_lacks_is_a_conflict() -> None:
    with pytest.raises(EvidenceConflict, match="unexpected handoff marker for slot 2"):
        observe(ALLOCATION, 42, 2, [], [Handoff(2, AT)], gated=True)


@pytest.mark.parametrize(
    ("rows", "markers", "expected"),
    [
        ([], [], None),
        ([QueueRow(42, "PENDING")], [], 42),
        ([], [Start(42, AT)], 42),
        ([], [End(42, 0, 1, AT), Stop(42, AT)], 42),
        ([QueueRow(42, "RUNNING")], [Start(42, AT)], 42),
        ([], [Handoff(0, AT)], None),
        ([], [Handoff(0, AT), Limit(42, AT)], 42),
    ],
)
def test_evidenced_job_is_the_one_job_any_evidence_names(
    rows: list[QueueRow], markers: list[Marker], expected: int | None
) -> None:
    assert evidenced_job(ALLOCATION, rows, markers) == expected


@pytest.mark.parametrize(
    ("rows", "markers", "shown"),
    [
        ([QueueRow(42, "RUNNING"), QueueRow(43, "PENDING")], [], "42, 43"),
        ([QueueRow(43, "RUNNING")], [Start(42, AT)], "42, 43"),
        ([], [Start(42, AT), Stop(44, AT)], "42, 44"),
    ],
)
def test_two_jobs_for_one_unresolved_identity_are_a_conflict(
    rows: list[QueueRow], markers: list[Marker], shown: str
) -> None:
    with pytest.raises(
        EvidenceConflict, match=f"allocation {ALLOCATION}: evidence names jobs {shown}"
    ):
        evidenced_job(ALLOCATION, rows, markers)
