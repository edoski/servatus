"""Property and metamorphic tests: scheduler replies are trusted only when they prove something.

Fail-closed means either the query raises (``EvidenceConflict`` or ``Unavailable``) or the
allocation's evidence is ``UNKNOWN`` and retained with a ``problem`` that withholds its Tasks.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scheduler_fixtures import (
    ALLOCATION,
    IDENTITY,
    MISSING,
    SLURM_BIN,
    SUBMIT,
    Scripted,
    query,
    sacct_row,
    squeue_row,
)

from servatus.campaign._evidence import (
    AllocationState,
    Expectation,
    SchedulerEvidence,
    parse_accounting,
    parse_active,
    parse_identity,
    parse_steps,
)
from servatus.campaign._remote import Completed
from servatus.campaign._scheduler import Scheduler
from servatus.errors import EvidenceConflict, ReconciliationError, Unavailable

EXPECTED = {42: Expectation(ALLOCATION, "2030-01-01T11:00:00", "2030-01-01T13:00:00", 2)}
ACTIVE = ["PENDING", "RUNNING", "COMPLETING", "EXPEDITING", "REQUEUE_HOLD"]
TERMINAL = ["COMPLETED", "FAILED", "TIMEOUT", "CANCELLED by 7", "OUT_OF_MEMORY"]
Reply = tuple[Completed, Completed]


def observe(squeue: Completed, sacct: Completed) -> SchedulerEvidence:
    transport = Scripted().queue("squeue", squeue).queue("sacct", sacct)
    return Scheduler(transport, SLURM_BIN).observe([query()])[ALLOCATION].allocation


def fails_closed(squeue: Completed, sacct: Completed) -> bool:
    try:
        result = observe(squeue, sacct)
    except (EvidenceConflict, Unavailable):
        return True
    return (
        result.state is AllocationState.UNKNOWN and result.retained and result.problem is not None
    )


def ok(text: str) -> Completed:
    return Completed(0, text.encode(), b"")


@st.composite
def accepted_replies(draw: st.DrawFn) -> Reply:
    """A consistent (squeue, sacct) pair, possibly with in-window requeue history."""
    requeued = draw(st.booleans())
    history = sacct_row("REQUEUED", submit="2030-01-01T11:30:00") if requeued else ""
    if draw(st.booleans()):
        state = draw(st.sampled_from(ACTIVE))
        accounting = draw(st.sampled_from(["", sacct_row(state)]))
        return ok(squeue_row(state)), ok(history + accounting if accounting else history)
    return MISSING, ok(history + sacct_row(draw(st.sampled_from(TERMINAL))))


def text(reply: Completed) -> str:
    return reply.stdout.decode()


Corruption = Callable[[Completed, Completed], Reply]
CORRUPTIONS: dict[str, Corruption] = {
    "foreign job in squeue": lambda q, a: (ok(text(q) + squeue_row(job=43)), a),
    "foreign job in sacct": lambda q, a: (q, ok(text(a) + sacct_row(job=43))),
    "foreign name in sacct": lambda q, a: (q, ok(text(a) + sacct_row(name="servatus-x"))),
    "queue comment missing": lambda q, a: (ok(squeue_row(comment="")), a),
    "duplicate squeue row": lambda q, a: (ok((text(q) or squeue_row()) * 2), a),
    "partial sacct": lambda q, a: (q, ok(text(a).rstrip("\n") or "42|x")),
    "partial squeue": lambda q, a: (ok((text(q) or squeue_row()).rstrip("\n")), a),
    "non-utf8 byte": lambda q, a: (
        (q, Completed(0, a.stdout.replace(b"None", b"\xff", 1), b""))
        if b"None" in a.stdout
        else (q, Completed(0, sacct_row().encode().replace(b"None", b"\xff"), b""))
    ),
    "squeue submitted before window": lambda q, a: (
        ok(squeue_row(submit="2029-12-31T23:00:00")),
        a,
    ),
    "squeue older than history": lambda q, a: (
        ok(squeue_row(submit="2030-01-01T11:10:00")),
        ok(sacct_row(submit="2030-01-01T11:20:00")),
    ),
    "malformed exit code": lambda q, a: (MISSING, ok(sacct_row("FAILED", exit_code="x:y"))),
    "non-canonical timestamp": lambda q, a: (MISSING, ok(sacct_row(submit="2030-1-01T12:00:00"))),
    "impossible date": lambda q, a: (MISSING, ok(sacct_row(submit="2030-02-30T12:00:00"))),
    "sacct stderr": lambda q, a: (q, Completed(0, a.stdout, b"sacct: warning\n")),
    "squeue failure": lambda q, a: (Completed(1, b"", b"squeue: comm failure\n"), a),
    "history before window": lambda q, a: (
        q,
        ok(sacct_row(submit="2030-01-01T10:00:00") + text(a)),
    ),
    "history out of order": lambda q, a: (
        q,
        ok(sacct_row(submit="2030-01-01T12:30:00") + sacct_row(submit="2030-01-01T12:10:00")),
    ),
    "contradictory terminal": lambda q, a: (
        ok(squeue_row("COMPLETED")),
        ok(sacct_row("FAILED")),
    ),
}


@settings(max_examples=400, deadline=None)
@given(accepted_replies(), st.sampled_from(sorted(CORRUPTIONS)))
def test_any_single_corruption_of_an_accepted_reply_fails_closed(
    reply: Reply, corruption: str
) -> None:
    squeue, sacct = reply
    clean = observe(squeue, sacct)  # precondition: the clean reply is trusted
    assert clean.problem is None and clean.state is not AllocationState.UNKNOWN
    assert fails_closed(*CORRUPTIONS[corruption](squeue, sacct)), corruption


@given(
    st.sampled_from(
        [
            Completed(1, b"", b"squeue: communication failure\n"),
            Completed(2, b"", MISSING.stderr),
            Completed(0, b"", MISSING.stderr),
            Completed(1, b"x\n", MISSING.stderr),
        ]
    )
)
def test_only_the_native_missing_reply_is_absence_even_with_healthy_accounting(
    squeue: Completed,
) -> None:
    try:
        observe(squeue, ok(sacct_row("COMPLETED")))
    except Unavailable:
        return
    raise AssertionError("squeue failure treated as absence")


FIELDS = st.sampled_from(
    [
        "42",
        "43",
        "0",
        "-1",
        "042",
        " 42",
        "alpha",
        "",
        "None",
        "Unknown",
        "N/A",
        IDENTITY,
        f" {IDENTITY}",
        f"{IDENTITY}-0",
        "42.0",
        "42.batch",
        "servatus-x",
        SUBMIT,
        "2030-01-01T12:30:00",
        "2030-01-01T15:00:00",
        "2029-12-31T23:00:00",
        "2030-02-30T00:00:00",
        "2030-1-1T00:00:00",
        "PENDING",
        "RUNNING",
        "COMPLETED",
        "FAILED",
        "REVOKED",
        "CANCELLED by 1",
        "TIMEOUT+",
        "FUTURE",
        "!bad",
        "0:0",
        "1:9",
        "x:y",
        "é",
        "a\tb",
    ]
)


def rows(width: int) -> st.SearchStrategy[bytes]:
    row = st.lists(FIELDS, min_size=width - 1, max_size=width + 1).map("|".join)
    lines = st.lists(row, max_size=4).map(lambda items: "".join(f"{x}\n" for x in items).encode())
    return lines | st.binary(max_size=64)


@settings(max_examples=2000, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(
    st.sampled_from([0, 1]),
    rows(8),
    rows(10),
    st.sampled_from([b"", b"warn", MISSING.stderr]),
)
def test_observation_only_ever_trusts_evidence_about_this_allocation(
    returncode: int, squeue: bytes, sacct: bytes, stderr: bytes
) -> None:
    try:
        result = observe(Completed(returncode, squeue, stderr), Completed(0, sacct, b""))
    except (EvidenceConflict, Unavailable):
        return
    assert result.state in set(AllocationState)
    if result.state is not AllocationState.UNKNOWN:
        assert IDENTITY.encode() in squeue + sacct
    if result.state.terminal:
        assert IDENTITY.encode() in sacct  # terminal evidence needs anchored accounting
    if result.problem is not None:
        assert (result.state, result.retained) == (AllocationState.UNKNOWN, True)


@settings(max_examples=1000)
@given(st.binary(max_size=256))
def test_row_parsers_raise_only_evidence_conflict_on_arbitrary_bytes(data: bytes) -> None:
    for parse in (
        lambda: parse_active(data, EXPECTED, None),
        lambda: parse_active(data, EXPECTED, "alpha"),
        lambda: parse_accounting(data, EXPECTED, None),
    ):
        with contextlib.suppress(EvidenceConflict):
            parse()
    try:
        steps = parse_steps(data, EXPECTED)
    except EvidenceConflict:
        return
    assert len(steps[42]) == 2


@settings(max_examples=1000)
@given(rows(4))
def test_step_parsing_yields_one_entry_per_slot_or_raises(output: bytes) -> None:
    try:
        steps = parse_steps(output, EXPECTED)
    except EvidenceConflict:
        return
    assert set(steps) == {42}
    assert len(steps[42]) == 2
    if any(item is not None for item in steps[42]):
        assert f"{IDENTITY}-".encode() in output


@settings(max_examples=2000)
@given(rows(3), rows(4))
def test_identity_proof_only_ever_fails_closed(squeue: bytes, sacct: bytes) -> None:
    try:
        job = parse_identity(squeue, sacct, IDENTITY)
    except ReconciliationError:
        return
    assert job.job_id > 0
    assert str(job.job_id).encode() in squeue + sacct
    assert IDENTITY.encode() in squeue + sacct
