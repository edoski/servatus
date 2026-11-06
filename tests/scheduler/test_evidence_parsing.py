from __future__ import annotations

from pathlib import PurePosixPath

import pytest
from scheduler_fixtures import ALLOCATION, IDENTITY, LOG_ROOT, allocation, listing

from servatus.campaign._evidence import (
    MAX_REPLY_LINES,
    End,
    Execution,
    Marker,
    Observation,
    QueueRow,
    Start,
    Stop,
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
        (f"{ALLOCATION}.42.start", Start(42)),
        (f"{ALLOCATION}.42.stop", Stop(42)),
        (f"{ALLOCATION}.42.0.0.end", End(42, 0, 0)),
        (f"{ALLOCATION}.42.17.3.end", End(42, 17, 3)),
        (f"{ALLOCATION}.9876543210.3.255.end", End(9876543210, 3, 255)),
    ],
)
def test_each_marker_kind_parses(name: str, marker: Marker) -> None:
    assert parse_markers(listing(name), ROOT, [ALLOCATION]) == {ALLOCATION: [marker]}


@pytest.mark.parametrize("code", range(256))
def test_every_exit_status_parses(code: int) -> None:
    markers = parse_markers(listing(f"{ALLOCATION}.42.1.{code}.end"), ROOT, [ALLOCATION])
    assert markers == {ALLOCATION: [End(42, 1, code)]}


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
    ],
)
def test_malformed_marker_of_a_queried_allocation_is_a_conflict(suffix: str) -> None:
    with pytest.raises(EvidenceConflict, match=f"allocation {ALLOCATION} has a malformed"):
        parse_markers(listing(f"{ALLOCATION}.{suffix}"), ROOT, [ALLOCATION])


def test_marker_name_that_is_not_ascii_is_a_conflict() -> None:
    output = f"{LOG_ROOT}/{ALLOCATION}.42.".encode() + b"\xff.start\n"
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
    assert parse_markers(output, ROOT, [ALLOCATION]) == {ALLOCATION: [Start(42)]}


def test_lines_outside_the_log_root_are_ignored() -> None:
    output = (
        listing(f"{ALLOCATION}.43.start", root="/elsewhere")
        + listing(f"{ALLOCATION}.43.start", root=f"{LOG_ROOT}2")
        + listing(f"sub/{ALLOCATION}.43.start")
        + listing(f"{ALLOCATION}.43.start", root=LOG_ROOT.removeprefix("/"))
        + listing(f"{ALLOCATION}.42.stop")
    )
    assert parse_markers(output, ROOT, [ALLOCATION]) == {ALLOCATION: [Stop(42)]}


@pytest.mark.parametrize("root", [LOG_ROOT, f"{LOG_ROOT}/", "/"])
def test_log_root_matches_with_or_without_a_trailing_slash(root: str) -> None:
    shown = root.rstrip("/")
    output = f"{shown}/{ALLOCATION}.42.start\n".encode()
    assert parse_markers(output, PurePosixPath(root), [ALLOCATION]) == {ALLOCATION: [Start(42)]}


def test_markers_are_grouped_by_allocation_in_listing_order() -> None:
    output = listing(
        f"{ALLOCATION}.42.1.0.end",
        f"{OTHER}.7.start",
        f"{ALLOCATION}.42.start",
        f"{ALLOCATION}.42.0.9.end",
    )
    assert parse_markers(output, ROOT, [ALLOCATION, OTHER, allocation(6)]) == {
        ALLOCATION: [End(42, 1, 0), Start(42), End(42, 0, 9)],
        OTHER: [Start(7)],
        allocation(6): [],
    }


def test_partial_marker_listing_is_a_conflict() -> None:
    with pytest.raises(EvidenceConflict, match="output is partial"):
        parse_markers(listing(f"{ALLOCATION}.42.start").rstrip(b"\n"), ROOT, [ALLOCATION])


# --- Observation -----------------------------------------------------------------------------


def observation(
    state: str | None, *, started: bool, stopped: bool, code: int | None
) -> Observation:
    return Observation(state, started, stopped, (code,))


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
    assert seen.execution(0) == (Execution.QUEUED, None)
    assert seen.live


@pytest.mark.parametrize("state", RUNNING_STATES + SETTLING_STATES)
@pytest.mark.parametrize(("started", "stopped"), [(False, False), (True, False), (True, True)])
def test_a_running_or_settling_row_follows_the_slot_end_marker(
    state: str, started: bool, stopped: bool
) -> None:
    def execution(code: int | None) -> tuple[Execution, int | None]:
        return observation(state, started=started, stopped=stopped, code=code).execution(0)

    assert execution(None) == (Execution.RUNNING, None)
    assert execution(0) == (Execution.SUCCEEDED, 0)
    assert execution(1) == (Execution.FAILED, 1)
    assert execution(255) == (Execution.FAILED, 255)
    assert observation(state, started=started, stopped=stopped, code=None).live


@pytest.mark.parametrize(("started", "stopped"), [(False, False), (True, False), (True, True)])
def test_a_purged_job_with_a_slot_end_marker_has_that_outcome(started: bool, stopped: bool) -> None:
    def execution(code: int) -> tuple[Execution, int | None]:
        return observation(None, started=started, stopped=stopped, code=code).execution(0)

    assert execution(0) == (Execution.SUCCEEDED, 0)
    assert execution(7) == (Execution.FAILED, 7)


@pytest.mark.parametrize(
    ("started", "stopped", "expected"),
    [
        (True, True, Execution.STOPPED),
        (False, False, Execution.STOPPED),
        (False, True, Execution.STOPPED),
        (True, False, Execution.LOST),
    ],
    ids=["interrupted", "never-started", "stopped-before-start", "lost"],
)
def test_a_purged_job_without_a_slot_end_marker(
    started: bool, stopped: bool, expected: Execution
) -> None:
    seen = observation(None, started=started, stopped=stopped, code=None)
    assert seen.execution(0) == (expected, None)
    assert not seen.live


def test_each_slot_is_decided_by_its_own_end_marker() -> None:
    seen = Observation(None, True, False, (0, None, 3))
    assert [seen.execution(slot) for slot in range(3)] == [
        (Execution.SUCCEEDED, 0),
        (Execution.LOST, None),
        (Execution.FAILED, 3),
    ]


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
    markers = [Start(42), End(42, 2, 0), End(42, 0, 5)]
    seen = observe(ALLOCATION, 42, 3, [QueueRow(42, "COMPLETING")], markers)
    assert seen == Observation("COMPLETING", True, False, (5, None, 0))
    assert observe(ALLOCATION, 42, 2, [], [Start(42), Stop(42)]) == Observation(
        None, True, True, (None, None)
    )
    assert observe(ALLOCATION, 42, 1, [], []) == Observation(None, False, False, (None,))


@pytest.mark.parametrize(
    ("rows", "markers", "message"),
    [
        ([QueueRow(43, "RUNNING")], [], "Slurm lists job 43 under its identity"),
        ([QueueRow(42, "RUNNING"), QueueRow(42, "RUNNING")], [], "lists job 42 twice"),
        ([], [Start(43)], "a run marker names job 43, not the accepted job 42"),
        ([], [Stop(43)], "a run marker names job 43"),
        ([], [End(43, 0, 0)], "a run marker names job 43"),
        ([], [End(42, 2, 0)], "unexpected end marker for slot 2"),
        ([], [End(42, 5, 0)], "unexpected end marker for slot 5"),
        ([], [End(42, 1, 0), End(42, 1, 1)], "unexpected end marker for slot 1"),
    ],
)
def test_observe_conflicts_name_the_allocation(
    rows: list[QueueRow], markers: list[Marker], message: str
) -> None:
    with pytest.raises(EvidenceConflict, match=f"allocation {ALLOCATION}: .*{message}"):
        observe(ALLOCATION, 42, 2, rows, markers)


@pytest.mark.parametrize(
    ("rows", "markers", "expected"),
    [
        ([], [], None),
        ([QueueRow(42, "PENDING")], [], 42),
        ([], [Start(42)], 42),
        ([], [End(42, 0, 1), Stop(42)], 42),
        ([QueueRow(42, "RUNNING")], [Start(42)], 42),
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
        ([QueueRow(43, "RUNNING")], [Start(42)], "42, 43"),
        ([], [Start(42), Stop(44)], "42, 44"),
    ],
)
def test_two_jobs_for_one_unresolved_identity_are_a_conflict(
    rows: list[QueueRow], markers: list[Marker], shown: str
) -> None:
    with pytest.raises(
        EvidenceConflict, match=f"allocation {ALLOCATION}: evidence names jobs {shown}"
    ):
        evidenced_job(ALLOCATION, rows, markers)
