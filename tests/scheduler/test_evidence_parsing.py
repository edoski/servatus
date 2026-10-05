from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from scheduler_fixtures import IDENTITY, SUBMIT, WINDOW, sacct_row, squeue_row

from servatus.campaign._evidence import (
    AllocationState,
    Expectation,
    JobRef,
    SchedulerEvidence,
    StepEvidence,
    combine,
    is_missing_reply,
    normalize_state,
    parse_accounting,
    parse_active,
    parse_identity,
    parse_receipt,
    parse_steps,
    submission_window,
)
from servatus.errors import ConfigurationError, EvidenceConflict, ReconciliationError

EXPECTED = {42: Expectation(IDENTITY.removeprefix("servatus-"), *WINDOW, 2)}


def evidence(squeue: str = "", sacct: str = "", *, cluster: str | None = None) -> SchedulerEvidence:
    active = parse_active(squeue.encode("utf-8", "surrogateescape"), EXPECTED, cluster)
    history = parse_accounting(sacct.encode("utf-8", "surrogateescape"), EXPECTED, cluster)
    return combine(EXPECTED[42], active[42], history[42])


# --- State table -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        *[
            (state, AllocationState.QUEUED)
            for state in (
                "PENDING",
                "CONFIGURING",
                "REQUEUED",
                "RESIZING",
                "SPECIAL_EXIT",
                "REQUEUE_HOLD",
                "REQUEUE_FED",
                "RESV_DEL_HOLD",
                "EXPEDITING",
            )
        ],
        *[
            (state, AllocationState.RUNNING)
            for state in ("RUNNING", "COMPLETING", "SIGNALING", "STAGE_OUT", "SUSPENDED", "STOPPED")
        ],
        ("COMPLETED", AllocationState.SUCCEEDED),
        ("CANCELLED", AllocationState.CANCELLED),
        ("PREEMPTED", AllocationState.CANCELLED),
        ("CANCELLED by 1234", AllocationState.CANCELLED),
        ("CANCELLED+", AllocationState.CANCELLED),
        *[
            (state, AllocationState.FAILED)
            for state in (
                "BOOT_FAIL",
                "DEADLINE",
                "FAILED",
                "NODE_FAIL",
                "OUT_OF_MEMORY",
                "TIMEOUT",
            )
        ],
        ("REVOKED", AllocationState.UNKNOWN),
        ("FUTURE_STATE", AllocationState.UNKNOWN),
        ("bad state!", AllocationState.UNKNOWN),
    ],
)
def test_one_state_table_normalizes_every_listed_state_exactly(
    raw: str, expected: AllocationState
) -> None:
    assert normalize_state(raw) is expected


def test_revoked_federation_sibling_is_unknown_and_retained_not_cancelled() -> None:
    # Regression: REVOKED (the job runs on another cluster) was treated as terminal CANCELLED,
    # which permitted a duplicate retry.
    result = evidence(sacct=sacct_row("REVOKED"))
    assert result.state is AllocationState.UNKNOWN
    assert result.raw_state == "REVOKED"
    assert result.retained


def test_expediting_is_queued_and_retained_from_either_source() -> None:
    assert evidence(sacct=sacct_row("EXPEDITING")).state is AllocationState.QUEUED
    assert evidence(sacct=sacct_row("EXPEDITING")).retained
    both = evidence(squeue_row("EXPEDITING"), sacct_row("EXPEDITING"))
    assert (both.state, both.retained, both.problem) == (AllocationState.QUEUED, True, None)


# --- Windows and receipts --------------------------------------------------------------------


def test_submission_window_is_utc_text_one_hour_either_side() -> None:
    zone = timezone(timedelta(hours=-10))
    intent = datetime(2030, 1, 1, 2, 0, 0, tzinfo=zone)
    assert submission_window(intent) == ("2030-01-01T11:00:00", "2030-01-01T13:00:00")
    with pytest.raises(ConfigurationError, match="timezone-aware"):
        submission_window(datetime(2030, 1, 1))


@pytest.mark.parametrize(
    ("output", "expected"),
    [(b"42\n", JobRef(42)), (b"42", JobRef(42)), (b"42;cluster-a\n", JobRef(42, "cluster-a"))],
)
def test_parse_receipt_accepts_one_exact_identity(output: bytes, expected: JobRef) -> None:
    assert parse_receipt(output) == expected


@pytest.mark.parametrize(
    "output",
    [b"0\n", b"-1\n", b"042\n", b"42;bad;extra\n", b"noise 42\n", b"", b"42\n\n", b"42;\n"],
)
def test_parse_receipt_rejects_unproved_identity(output: bytes) -> None:
    with pytest.raises(EvidenceConflict, match="exactly one job identity"):
        parse_receipt(output)


# --- Missing-job reply -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "cluster", "missing"),
    [
        (1, b"", b"slurm_load_jobs error: Invalid job id specified\n", None, True),
        (
            1,
            b"CLUSTER: alpha\n",
            b"slurm_load_jobs error: Invalid job id specified\n",
            "alpha",
            True,
        ),
        (
            1,
            b"CLUSTER: beta\n",
            b"slurm_load_jobs error: Invalid job id specified\n",
            "alpha",
            False,
        ),
        (2, b"", b"slurm_load_jobs error: Invalid job id specified\n", None, False),
        (1, b"partial", b"slurm_load_jobs error: Invalid job id specified\n", None, False),
        (1, b"", b"squeue: communication failure\n", None, False),
        (0, b"", b"slurm_load_jobs error: Invalid job id specified\n", None, False),
    ],
)
def test_only_the_exact_native_missing_reply_means_absence(
    returncode: int, stdout: bytes, stderr: bytes, cluster: str | None, missing: bool
) -> None:
    assert is_missing_reply(returncode, stdout, stderr, cluster) is missing


# --- Identity guards (raise for the whole reply) ---------------------------------------------


@pytest.mark.parametrize(
    ("squeue", "sacct", "message"),
    [
        (squeue_row(job=43), "", "unrelated job"),
        (squeue_row(job="042"), "", "unrelated job"),
        (squeue_row(job=" 42"), "", "unrelated job"),
        (squeue_row(name="servatus-000000000000000000000000"), "", "unrelated allocation"),
        (squeue_row(comment=""), "", "unrelated allocation"),
        (squeue_row(comment=f" {IDENTITY} "), "", "unrelated allocation"),
        ("", sacct_row(job=43), "unrelated job"),
        ("", sacct_row(name=f" {IDENTITY} "), "unrelated allocation"),
        ("", sacct_row(comment="wrong-identity"), "unrelated allocation"),
        ("", sacct_row(comment=" N/A "), "unrelated allocation"),
        ("", sacct_row(comment=" "), "unrelated allocation"),
        ("", sacct_row(cluster="bad cluster"), "unrelated cluster"),
    ],
)
def test_identity_violations_raise_evidence_conflict(squeue: str, sacct: str, message: str) -> None:
    with pytest.raises(EvidenceConflict, match=message):
        evidence(squeue, sacct)


def test_accounting_cluster_must_match_the_queried_route() -> None:
    assert evidence(sacct=sacct_row(cluster="alpha"), cluster="alpha").state is (
        AllocationState.RUNNING
    )
    with pytest.raises(EvidenceConflict, match="unrelated cluster"):
        evidence(sacct=sacct_row(cluster="beta"), cluster="alpha")
    with pytest.raises(EvidenceConflict, match="unrelated cluster"):
        evidence(sacct=sacct_row(cluster=""), cluster="alpha")


@pytest.mark.parametrize("comment", ["", "None", "Unknown", "N/A", IDENTITY])
def test_accounting_may_omit_the_comment_but_the_queue_may_not(comment: str) -> None:
    assert evidence(sacct=sacct_row("PENDING", comment=comment)).state is AllocationState.QUEUED


@pytest.mark.parametrize(
    ("squeue", "sacct", "message"),
    [
        (squeue_row() + "x", "", "partial"),
        ("42|RUNNING\n", "", "malformed"),
        (squeue_row().replace("None", "\udcff"), "", "malformed"),
        ("", sacct_row(exit_code="x:y"), "exit code"),
        ("", sacct_row(submit="2030-01-01T1:00:00"), "timestamp"),
        ("", sacct_row(submit="2030-02-30T01:00:00"), "timestamp"),
        ("", sacct_row(submit="Unknown"), "submit time"),
        ("", sacct_row(start="2030-01-01 12:00:00"), "timestamp"),
        ("", sacct_row(state="!bad"), "state"),
        ("", sacct_row(identity=IDENTITY, reason="x" * 4097), "malformed"),
    ],
)
def test_malformed_replies_raise_evidence_conflict(squeue: str, sacct: str, message: str) -> None:
    with pytest.raises(EvidenceConflict, match=message):
        evidence(squeue, sacct)


@pytest.mark.parametrize("control", ["\t", "\r", "\x00", "\x7f"])
def test_fields_reject_controls_before_space_normalization(control: str) -> None:
    with pytest.raises(EvidenceConflict, match="malformed"):
        evidence(squeue_row(f" RUNNING{control}"))


def test_reply_line_bound_is_enforced() -> None:
    with pytest.raises(EvidenceConflict, match="line bound"):
        evidence(sacct=sacct_row() * 4097)


def test_padded_state_and_time_fields_are_normalized() -> None:
    row = squeue_row(" RUNNING ", submit=f" {SUBMIT} ", start=" 2030-01-01T12:01:00 ")
    result = evidence(row)
    assert (result.state, result.raw_state, result.started_at) == (
        AllocationState.RUNNING,
        "RUNNING",
        "2030-01-01T12:01:00",
    )


# --- Combination -----------------------------------------------------------------------------


def test_nothing_observed_is_unknown_and_not_retained() -> None:
    assert evidence() == SchedulerEvidence(AllocationState.UNKNOWN)


def test_queue_evidence_alone_proves_retained_work() -> None:
    result = evidence(squeue_row("PENDING", reason="Priority"))
    assert (result.state, result.retained, result.reason) == (
        AllocationState.QUEUED,
        True,
        "Priority",
    )
    unknown = evidence(squeue_row("FUTURE_RETAINED_STATE"))
    assert (unknown.state, unknown.retained) == (AllocationState.UNKNOWN, True)


def test_terminal_queue_row_without_accounting_anchor_is_unknown() -> None:
    result = evidence(squeue_row("COMPLETED"))
    assert (result.state, result.retained, result.raw_state) == (
        AllocationState.UNKNOWN,
        False,
        "COMPLETED",
    )


def test_terminal_accounting_carries_exit_code_times_and_reason() -> None:
    row = sacct_row(
        "FAILED",
        exit_code="1:0",
        reason="NonZeroExitCode",
        start="2030-01-01T12:01:00",
        end="2030-01-01T13:30:00",
    )
    assert evidence(sacct=row) == SchedulerEvidence(
        AllocationState.FAILED,
        raw_state="FAILED",
        accounting_state="FAILED",
        exit_code="1:0",
        reason="NonZeroExitCode",
        started_at="2030-01-01T12:01:00",
        ended_at="2030-01-01T13:30:00",
    )


@pytest.mark.parametrize(
    ("queue", "accounting", "state", "raw", "retained"),
    [
        # Regression (lag): squeue already terminal, slurmdbd still reports an older state.
        ("COMPLETED", "RUNNING", AllocationState.SUCCEEDED, "COMPLETED", True),
        ("FAILED", "PENDING", AllocationState.FAILED, "FAILED", True),
        ("RUNNING", "COMPLETED", AllocationState.SUCCEEDED, "COMPLETED", True),
        ("RUNNING", "PENDING", AllocationState.RUNNING, "RUNNING", True),
        ("PENDING", "RUNNING", AllocationState.RUNNING, "RUNNING", True),
        ("COMPLETING", "RUNNING", AllocationState.RUNNING, "COMPLETING", True),
        ("CANCELLED by 1234", "CANCELLED+", AllocationState.CANCELLED, "CANCELLED by 1234", False),
        ("FAILED", "TIMEOUT", AllocationState.FAILED, "TIMEOUT", False),
        ("SPECIAL_EXIT", "COMPLETED", AllocationState.SUCCEEDED, "COMPLETED", True),
    ],
)
def test_same_incarnation_prefers_the_more_advanced_state_and_stays_conservative(
    queue: str, accounting: str, state: AllocationState, raw: str, retained: bool
) -> None:
    result = evidence(squeue_row(queue), sacct_row(accounting))
    assert (result.state, result.raw_state, result.retained, result.problem) == (
        state,
        raw,
        retained,
        None,
    )
    assert result.accounting_state == accounting


@pytest.mark.parametrize(
    ("queue", "accounting"),
    [
        ("COMPLETED", "FAILED"),
        ("CANCELLED", "COMPLETED"),
        ("FUTURE_ACTIVE", "FUTURE_ACCOUNTING"),
        ("REVOKED", "RUNNING"),
    ],
)
def test_irreconcilable_same_incarnation_is_a_contained_problem(
    queue: str, accounting: str
) -> None:
    result = evidence(squeue_row(queue), sacct_row(accounting))
    assert (result.state, result.retained) == (AllocationState.UNKNOWN, True)
    assert result.problem == f"squeue reports {queue} but sacct reports {accounting}"
    assert (result.raw_state, result.accounting_state) == (queue, accounting)


def test_requeue_inside_the_submission_window_is_accepted() -> None:
    # Regression (wedge): a requeue 20 minutes after submission put a second accounting row in
    # the window, which made inspection fail permanently.
    history = sacct_row("REQUEUED", submit="2030-01-01T12:00:00") + sacct_row(
        "RUNNING", submit="2030-01-01T12:20:00"
    )
    running = evidence(squeue_row("RUNNING", submit="2030-01-01T12:20:00"), history)
    assert (running.state, running.retained, running.problem) == (
        AllocationState.RUNNING,
        True,
        None,
    )
    finished = sacct_row("REQUEUED", submit="2030-01-01T12:00:00") + sacct_row(
        "COMPLETED", submit="2030-01-01T12:20:00", end="2030-01-02T00:00:00"
    )
    done = evidence(sacct=finished)
    assert (done.state, done.retained, done.ended_at) == (
        AllocationState.SUCCEEDED,
        False,
        "2030-01-02T00:00:00",
    )


def test_requeue_after_the_window_uses_the_latest_incarnation() -> None:
    history = sacct_row("RUNNING", submit="2030-01-01T12:00:00") + sacct_row(
        "PENDING", submit="2030-01-01T14:00:00"
    )
    active = evidence(squeue_row("RUNNING", submit="2030-01-01T15:00:00"), history)
    assert (active.state, active.accounting_state, active.retained) == (
        AllocationState.RUNNING,
        "PENDING",
        True,
    )
    terminal = sacct_row("RUNNING", submit="2030-01-01T12:00:00") + sacct_row(
        "COMPLETED", submit="2030-01-01T14:00:00", end="2030-01-01T16:00:00"
    )
    assert evidence(sacct=terminal).state is AllocationState.SUCCEEDED


def test_queue_row_sampled_before_a_recorded_requeue_defers_to_accounting() -> None:
    history = sacct_row("REQUEUED", submit="2030-01-01T12:00:00") + sacct_row(
        "PENDING", submit="2030-01-01T12:30:00"
    )
    result = evidence(squeue_row("RUNNING", submit="2030-01-01T12:00:00"), history)
    assert (result.state, result.retained, result.problem) == (AllocationState.QUEUED, True, None)


@pytest.mark.parametrize(
    "rows",
    [
        (("2030-01-01T14:00:00", "RUNNING"),),
        (("2030-01-01T10:00:00", "RUNNING"), ("2030-01-01T12:00:00", "RUNNING")),
        (
            ("2030-01-01T12:00:00", "RUNNING"),
            ("2030-01-01T15:00:00", "PENDING"),
            ("2030-01-01T14:00:00", "RUNNING"),
        ),
        (("2030-01-01T12:00:00", "RUNNING"), ("2030-01-01T12:00:00", "PENDING")),
    ],
)
def test_unanchored_or_unordered_history_is_a_contained_problem(
    rows: tuple[tuple[str, str], ...],
) -> None:
    sacct = "".join(sacct_row(state, submit=submit) for submit, state in rows)
    result = evidence(sacct=sacct)
    assert (result.state, result.retained) == (AllocationState.UNKNOWN, True)
    assert result.problem is not None and "accounting" in result.problem


def test_history_from_two_clusters_is_a_contained_problem() -> None:
    sacct = sacct_row(cluster="alpha") + sacct_row(cluster="beta", submit="2030-01-01T12:30:00")
    assert evidence(sacct=sacct).problem == "accounting requeue history is not ordered"


@pytest.mark.parametrize(
    ("squeue", "sacct"),
    [
        (squeue_row(submit="2029-12-31T23:00:00"), ""),
        (squeue_row(submit="2030-01-01T11:30:00"), sacct_row(submit="2030-01-01T12:00:00")),
    ],
)
def test_queue_row_older_than_the_allocation_is_a_contained_problem(
    squeue: str, sacct: str
) -> None:
    result = evidence(squeue, sacct)
    assert result.problem == "queue row predates the allocation's submission"
    assert (result.state, result.retained) == (AllocationState.UNKNOWN, True)


def test_two_queue_rows_for_one_job_are_a_contained_problem() -> None:
    result = evidence(squeue_row() + squeue_row("PENDING"))
    assert result.problem == "squeue returned several rows for one job"


@pytest.mark.parametrize("raw", ["RUNNING", "SPECIAL_EXIT", "REQUEUE_HOLD", "RESV_DEL_HOLD"])
@pytest.mark.parametrize("accounting", [False, True])
def test_retained_queue_work_blocks_even_after_terminal_accounting(
    raw: str, accounting: bool
) -> None:
    sacct = sacct_row("COMPLETED") if accounting else ""
    assert evidence(squeue_row(raw), sacct).retained


# --- Steps -----------------------------------------------------------------------------------


def test_step_rows_map_to_slots_by_name_and_ignore_everything_else() -> None:
    output = (
        f"42|{IDENTITY}|RUNNING|0:0\n"
        "42.batch|batch|RUNNING|0:0\n"
        "42.extern|extern|RUNNING|0:0\n"
        f"42.0|{IDENTITY}-1|FAILED|1:0\n"
        f"42.1|{IDENTITY}-0|COMPLETED|0:0\n"
        f"42.2|{IDENTITY}-2|COMPLETED|0:0\n"
        f"42.3|{IDENTITY}-01|COMPLETED|0:0\n"
        "42.4|servatus-ffffffffffffffffffffffff-0|COMPLETED|0:0\n"
    )
    assert parse_steps(output.encode(), EXPECTED) == {
        42: (
            StepEvidence(AllocationState.SUCCEEDED, "COMPLETED", "0:0"),
            StepEvidence(AllocationState.FAILED, "FAILED", "1:0"),
        )
    }


def test_a_slot_reported_twice_has_no_step_evidence() -> None:
    output = f"42.0|{IDENTITY}-0|COMPLETED|0:0\n42.1|{IDENTITY}-0|FAILED|1:0\n"
    assert parse_steps(output.encode(), EXPECTED) == {42: (None, None)}


def test_malformed_step_rows_raise_for_the_caller_to_degrade() -> None:
    with pytest.raises(EvidenceConflict, match="exit code"):
        parse_steps(f"42.0|{IDENTITY}-0|FAILED|bad\n".encode(), EXPECTED)


# --- Identity proof for reconciliation -------------------------------------------------------


def test_identity_accepts_absent_accounting_comment_and_adopts_cluster() -> None:
    squeue = f"42|{IDENTITY}|{IDENTITY}\n".encode()
    sacct = f"42|{IDENTITY}||alpha\n42|{IDENTITY}|{IDENTITY}|alpha\n".encode()
    assert parse_identity(squeue, sacct, IDENTITY) == JobRef(42, "alpha")
    assert parse_identity(b"", f"42|{IDENTITY}|N/A|\n".encode(), IDENTITY) == JobRef(42)


@pytest.mark.parametrize(
    ("squeue", "sacct", "message"),
    [
        ("", "", "0 jobs"),
        (f"42|{IDENTITY}|{IDENTITY}\n", f"43|{IDENTITY}|{IDENTITY}|alpha\n", "2 jobs"),
        ("42|wrong|wrong\n", "", "unrelated"),
        (f"42|{IDENTITY}|{IDENTITY}\n", f"42|{IDENTITY}|wrong-identity|alpha\n", "unrelated"),
        (f"42|{IDENTITY}|{IDENTITY}\nmalformed\n", "", "malformed"),
        (f"42|{IDENTITY}|{IDENTITY}\n0|{IDENTITY}|{IDENTITY}\n", "", "invalid job"),
        (f"42|{IDENTITY}|{IDENTITY}\n", f"42|{IDENTITY}||alpha\nmalformed\n", "malformed"),
        (f"42| {IDENTITY} |{IDENTITY}\n", "", "unrelated"),
        ("", f"42|{IDENTITY}| N/A |alpha\n", "unrelated"),
        (f"42|{IDENTITY}|{IDENTITY}\n", f"42|{IDENTITY}||bad cluster\n", "invalid cluster"),
        ("", f"42|{IDENTITY}||alpha\n42|{IDENTITY}||beta\n", "conflicting cluster"),
        (f"42|{IDENTITY}|{IDENTITY}", "", "partial"),
    ],
)
def test_identity_that_is_not_exactly_one_proven_job_is_a_reconciliation_error(
    squeue: str, sacct: str, message: str
) -> None:
    with pytest.raises(ReconciliationError, match=message):
        parse_identity(squeue.encode(), sacct.encode(), IDENTITY)


def test_window_and_submit_constants_agree() -> None:
    assert submission_window(datetime(2030, 1, 1, 12, tzinfo=UTC)) == WINDOW
    assert WINDOW[0] <= SUBMIT <= WINDOW[1]
