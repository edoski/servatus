from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath

import pytest
from scheduler_fixtures import (
    ALLOCATION,
    IDENTITY,
    INTENT,
    MISSING,
    SLURM_BIN,
    Scripted,
    allocation,
    ok,
    option,
    query,
    sacct_row,
    squeue_row,
)

from servatus.campaign._evidence import AllocationState, JobRef, StepEvidence
from servatus.campaign._remote import Completed
from servatus.campaign._scheduler import Scheduler
from servatus.errors import (
    ConfigurationError,
    EvidenceConflict,
    ReconciliationError,
    Unavailable,
)

SBATCH = f"{SLURM_BIN}/sbatch"
ARGV = (SBATCH, "--parsable", "--export=NIL", f"--job-name={IDENTITY}")
SACCT_FORMAT = (
    "--format=JobIDRaw%64,Cluster%256,JobName%256,Comment%256,Submit%32,State%256,ExitCode%32,"
    "Reason%4096,Start%32,End%32"
)


def scheduler(transport: Scripted) -> Scheduler:
    return Scheduler(transport, SLURM_BIN)


# --- observe ---------------------------------------------------------------------------------


def test_observe_issues_exact_local_queries_and_combines_rows() -> None:
    steps = f"42.0|{IDENTITY}-0|RUNNING|0:0\n"
    transport = (
        Scripted()
        .queue("squeue", ok(squeue_row()))
        .queue("sacct", ok(sacct_row("PENDING")), ok(steps))
    )
    observed = scheduler(transport).observe([query(task_count=2)])
    assert transport.calls == [
        (
            f"{SLURM_BIN}/squeue",
            "--noheader",
            "--jobs",
            "42",
            "--local",
            "--format=%i|%j|%k|%V|%T|%r|%S|%e",
        ),
        (
            f"{SLURM_BIN}/sacct",
            "--noheader",
            "--parsable2",
            "--allocations",
            "--duplicates",
            "--jobs",
            "42",
            "--name",
            IDENTITY,
            "--starttime",
            "2030-01-01T11:00:00",
            "--local",
            SACCT_FORMAT,
        ),
        (
            f"{SLURM_BIN}/sacct",
            "--noheader",
            "--parsable2",
            "--jobs",
            "42",
            "--starttime",
            "2030-01-01T11:00:00",
            "--local",
            "--format=JobIDRaw%64,JobName%256,State%256,ExitCode%32",
        ),
    ]
    result = observed[ALLOCATION]
    assert result.allocation.state is AllocationState.RUNNING
    assert result.allocation.accounting_state == "PENDING"
    assert result.steps == (StepEvidence(AllocationState.RUNNING, "RUNNING", "0:0"), None)


def test_observe_routes_by_cluster_and_tolerates_the_cluster_banner() -> None:
    transport = (
        Scripted()
        .queue("squeue", ok("CLUSTER: alpha\n" + squeue_row()))
        .queue("sacct", ok(sacct_row(cluster="alpha")))
    )
    observed = scheduler(transport).observe([query(cluster="alpha")])
    assert all("--clusters=alpha" in call and "--local" not in call for call in transport.calls)
    assert observed[ALLOCATION].allocation.state is AllocationState.RUNNING


def test_native_missing_queue_reply_still_uses_exact_terminal_accounting() -> None:
    transport = (
        Scripted()
        .queue("squeue", MISSING)
        .queue("sacct", ok(sacct_row("COMPLETED", end="2030-01-01T13:00:00")))
    )
    result = scheduler(transport).observe([query()])[ALLOCATION].allocation
    assert (result.state, result.retained) == (AllocationState.SUCCEEDED, False)


@pytest.mark.parametrize(
    "failure",
    [
        Completed(2, b"", MISSING.stderr),
        Completed(1, b"partial", MISSING.stderr),
        Completed(1, b"", b"squeue: communication failure\n"),
        Completed(0, b"", MISSING.stderr),
        Completed(0, squeue_row().encode(), b"warning\n"),
    ],
)
def test_any_other_queue_failure_is_unavailable(failure: Completed) -> None:
    transport = Scripted().queue("squeue", failure)
    with pytest.raises(Unavailable, match="squeue failed"):
        scheduler(transport).observe([query()])
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "failure", [Completed(1, b"", b"sacct: error\n"), Completed(0, b"", b"sacct: warning\n")]
)
def test_accounting_failure_is_unavailable(failure: Completed) -> None:
    transport = Scripted().queue("squeue", MISSING).queue("sacct", failure)
    with pytest.raises(Unavailable, match="sacct failed with exit status"):
        scheduler(transport).observe([query()])


def test_transport_failure_propagates_as_unavailable() -> None:
    transport = Scripted().queue("squeue", Unavailable("ssh: connection refused"))
    with pytest.raises(Unavailable, match="connection refused"):
        scheduler(transport).observe([query()])


def test_unrelated_rows_abort_the_batch_with_evidence_conflict() -> None:
    transport = Scripted().queue("squeue", ok(squeue_row(job=43)))
    with pytest.raises(EvidenceConflict, match="unrelated job"):
        scheduler(transport).observe([query()])


def test_one_contradictory_attempt_cannot_hide_the_rest_of_its_batch() -> None:
    # Regression (wedge): one requeued job used to abort observation of all 16 in its batch.
    queries = [query(100 + index, allocation_id=allocation(index)) for index in range(16)]

    def history(argv: tuple[str, ...]) -> Completed:
        rows = ""
        for index, item in enumerate(queries):
            identity = f"servatus-{item.allocation_id}"
            rows += sacct_row("COMPLETED", job=item.job.job_id, identity=identity)
            if index == 3:
                rows += sacct_row("COMPLETED", job=item.job.job_id, identity=identity)
        return ok(rows)

    transport = Scripted().queue("squeue", MISSING).queue("sacct", history)
    observed = scheduler(transport).observe(queries)
    problems = {key for key, value in observed.items() if value.allocation.problem is not None}
    assert problems == {allocation(3)}
    assert observed[allocation(3)].allocation.retained
    assert all(
        value.allocation.state is AllocationState.SUCCEEDED
        for key, value in observed.items()
        if key != allocation(3)
    )


def test_observe_batches_by_cluster_bound_and_reused_job_numbers() -> None:
    queries = [
        query(index + 1, cluster="alpha" if index % 2 else "beta", allocation_id=allocation(index))
        for index in range(35)
    ]
    queries.append(query(2, cluster="alpha", allocation_id=allocation(99)))  # reused job number
    transport = Scripted()
    observed = scheduler(transport).observe(queries)
    assert list(observed) == [item.allocation_id for item in queries]
    squeues = transport.named("squeue")
    batches = [option(call, "--jobs").split(",") for call in squeues]
    assert all(1 <= len(jobs) <= 16 for jobs in batches)
    assert all(len(set(jobs)) == len(jobs) for jobs in batches)
    for call, jobs in zip(squeues, batches, strict=True):
        cluster = "alpha" if int(jobs[0]) % 2 == 0 else "beta"
        assert f"--clusters={cluster}" in call
        assert {int(job) % 2 for job in jobs} == {int(jobs[0]) % 2}
    assert sorted(len(jobs) for jobs in batches) == [2, 2, 16, 16]
    assert sum(len(jobs) for jobs in batches) == 36
    assert len(transport.calls) == 3 * len(batches)
    assert all(value.allocation.state is AllocationState.UNKNOWN for value in observed.values())


def test_observe_matches_rows_independently_of_reply_order() -> None:
    queries = [query(100 + index, allocation_id=allocation(index)) for index in range(3)]
    states = ["FAILED", "RUNNING", "COMPLETED"]
    rows = [
        sacct_row(state, job=item.job.job_id, identity=f"servatus-{item.allocation_id}")
        for item, state in zip(queries, states, strict=True)
    ]
    transport = Scripted().queue("squeue", ok("")).queue("sacct", ok("".join(reversed(rows))))
    observed = scheduler(transport).observe(queries)
    assert [observed[item.allocation_id].allocation.state for item in queries] == [
        AllocationState.FAILED,
        AllocationState.RUNNING,
        AllocationState.SUCCEEDED,
    ]
    assert option(transport.calls[1], "--name") == ",".join(
        f"servatus-{item.allocation_id}" for item in queries
    )


def test_accounting_window_starts_at_the_earliest_intent_in_the_batch() -> None:
    queries = [
        query(1, allocation_id=allocation(1), intent_at=INTENT),
        query(2, allocation_id=allocation(2), intent_at=INTENT - timedelta(days=1)),
    ]
    transport = Scripted()
    scheduler(transport).observe(queries)
    assert option(transport.calls[1], "--starttime") == "2029-12-31T11:00:00"


@pytest.mark.parametrize(
    "steps",
    [
        Completed(1, b"", b"sacct: error\n"),
        Completed(0, b"", b"sacct: warning\n"),
        Completed(0, b"42.0|malformed\n", b""),
        Unavailable("timeout"),
    ],
)
def test_step_query_failures_make_step_evidence_unavailable(steps: Completed | Unavailable) -> None:
    transport = Scripted().queue("squeue", MISSING).queue("sacct", ok(sacct_row("RUNNING")), steps)
    observed = scheduler(transport).observe([query(task_count=3)])[ALLOCATION]
    assert observed.allocation.state is AllocationState.RUNNING
    assert observed.steps == ()


def test_observe_rejects_duplicate_allocations_and_foreign_values() -> None:
    with pytest.raises(ConfigurationError, match="once per query"):
        scheduler(Scripted()).observe([query(1), query(2)])
    with pytest.raises(ConfigurationError, match="AttemptQuery"):
        scheduler(Scripted()).observe(["not a query"])  # pyright: ignore[reportArgumentType]


def test_observe_of_nothing_issues_no_commands() -> None:
    transport = Scripted()
    assert scheduler(transport).observe([]) == {}
    assert transport.calls == []


# --- submit, test_only, ping -----------------------------------------------------------------


def test_submit_streams_the_exact_script_and_parses_the_receipt() -> None:
    transport = Scripted().queue("sbatch", ok("42;alpha\n"))
    script = b"#!/bin/sh\n\x00\xff"
    assert scheduler(transport).submit(ARGV, script) == JobRef(42, "alpha")
    assert transport.calls == [ARGV]
    assert transport.stdins == [script]


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (
            Completed(1, b"", b"sbatch: error: Invalid account\n"),
            "status 1: sbatch: error: Invalid",
        ),
        (Completed(0, b"Submitted batch job 42\n", b""), "exactly one job identity"),
        (Completed(0, b"", b""), "exactly one job identity"),
        (Unavailable("deadline"), "deadline"),
    ],
)
def test_submit_failures_are_unavailable(reply: Completed | Unavailable, message: str) -> None:
    with pytest.raises(Unavailable, match=message):
        scheduler(Scripted().queue("sbatch", reply)).submit(ARGV, b"#!/bin/sh\n")


@pytest.mark.parametrize(
    "argv",
    [
        ("/usr/bin/sbatch", "--parsable"),
        (),
        (*ARGV, "--test-only"),
    ],
)
def test_submit_refuses_foreign_argv_before_contact(argv: tuple[str, ...]) -> None:
    transport = Scripted()
    with pytest.raises(ConfigurationError, match="this target's sbatch"):
        scheduler(transport).submit(argv, b"")
    assert transport.calls == []


def test_command_bounds_are_configuration_errors_before_contact() -> None:
    transport = Scripted()
    with pytest.raises(ConfigurationError, match="bound of 4096 bytes"):
        scheduler(transport).submit((*ARGV, "--comment=" + "x" * 4096), b"")
    assert transport.calls == []


def test_test_only_appends_the_flag_and_reports_both_streams() -> None:
    reply = Completed(1, b"out\n", b"sbatch: error: Requested node configuration\n")
    transport = Scripted().queue("sbatch", ok(""), reply)
    accepted = scheduler(transport).test_only(ARGV, b"#!/bin/sh\n")
    assert accepted == (True, "", "")
    assert transport.calls[0] == (*ARGV, "--test-only")
    assert scheduler(transport).test_only(ARGV, b"#!/bin/sh\n") == (
        False,
        "out",
        "sbatch: error: Requested node configuration",
    )


def test_ping_reports_the_slurm_version() -> None:
    transport = Scripted().queue("sbatch", ok("slurm 23.11.4\n"), Completed(127, b"", b"no"))
    assert scheduler(transport).ping() == "slurm 23.11.4"
    assert transport.calls == [(SBATCH, "--version")]
    with pytest.raises(Unavailable, match="exit status 127: no"):
        scheduler(transport).ping()


def test_scheduler_requires_an_absolute_slurm_bin() -> None:
    with pytest.raises(ConfigurationError, match="slurm_bin"):
        Scheduler(Scripted(), "opt/slurm/bin")


# --- identify --------------------------------------------------------------------------------


def test_identify_queries_queue_and_windowed_accounting() -> None:
    transport = (
        Scripted()
        .queue("squeue", ok(f"42|{IDENTITY}|{IDENTITY}\n"))
        .queue("sacct", ok(f"42|{IDENTITY}||alpha\n"))
    )
    intent = datetime(2030, 1, 1, 2, 0, 0, tzinfo=timezone(timedelta(hours=-10)))
    assert scheduler(transport).identify(ALLOCATION, intent) == JobRef(42, "alpha")
    assert transport.calls == [
        (f"{SLURM_BIN}/squeue", "--noheader", "--name", IDENTITY, "--format=%i|%j|%k"),
        (
            f"{SLURM_BIN}/sacct",
            "--noheader",
            "--parsable2",
            "--allocations",
            "--duplicates",
            "--name",
            IDENTITY,
            "--starttime",
            "2030-01-01T11:00:00",
            "--endtime",
            "2030-01-01T13:00:00",
            "--format=JobIDRaw,JobName,Comment,Cluster",
        ),
    ]


def test_identify_without_exactly_one_job_is_a_reconciliation_error() -> None:
    transport = Scripted().queue("squeue", ok("")).queue("sacct", ok(""))
    with pytest.raises(ReconciliationError, match="0 jobs"):
        scheduler(transport).identify(ALLOCATION, INTENT)


def test_identify_command_failure_is_unavailable() -> None:
    transport = Scripted().queue("squeue", ok("")).queue("sacct", Completed(1, b"", b"down\n"))
    with pytest.raises(Unavailable, match="sacct failed with exit status 1: down"):
        scheduler(transport).identify(ALLOCATION, INTENT)


# --- tail ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (b"", (b"", False)),
        (b"\x00\xff", (b"\x00\xff", False)),
        (b"abcd", (b"abcd", False)),
        (b"Xabcd", (b"abcd", True)),
    ],
)
def test_tail_returns_an_exact_bounded_binary_suffix(
    reply: bytes, expected: tuple[bytes, bool]
) -> None:
    transport = Scripted().queue("tail", ok(reply))
    path = PurePosixPath(f"/cluster/logs/{ALLOCATION}-42.out")
    assert scheduler(transport).tail(path, 4) == expected
    assert transport.calls == [("/usr/bin/tail", "-c", "5", "--", str(path))]
    assert transport.limits == [5]


@pytest.mark.parametrize(
    "outcome",
    [
        Completed(1, b"partial-content", b"remote-stderr"),
        Completed(0, b"partial-content", b"remote-stderr"),
        Completed(0, b"overflowing", b""),
        Unavailable("secret-host: secret-command partial-content"),
    ],
)
def test_tail_failures_are_redacted_and_never_return_partial_content(
    outcome: Completed | Unavailable,
) -> None:
    transport = Scripted().queue("tail", outcome)
    with pytest.raises(Unavailable) as caught:
        scheduler(transport).tail(PurePosixPath("/secret/log/path"), 4)
    rendered: list[str] = []
    error: BaseException | None = caught.value
    while error is not None:
        rendered.append(str(error))
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    assert rendered == ["log is unavailable"]


@pytest.mark.parametrize("maximum", [0, -1, True, 1.5, "1", 1024 * 1024 + 1])
def test_tail_byte_limit_is_a_positive_mib_or_less(maximum: object) -> None:
    with pytest.raises(ConfigurationError, match="max_bytes"):
        scheduler(Scripted()).tail(PurePosixPath("/log"), maximum)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("path", ["relative/log", "/log\nINJECT"])
def test_tail_path_must_be_absolute_and_printable(path: str) -> None:
    with pytest.raises(ConfigurationError, match="absolute POSIX path"):
        scheduler(Scripted()).tail(PurePosixPath(path), 4)


# --- cancel ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("job", "expected"),
    [
        (JobRef(42), (f"{SLURM_BIN}/scancel", f"--name={IDENTITY}", "42")),
        (
            JobRef(42, "alpha"),
            (f"{SLURM_BIN}/scancel", "--clusters=alpha", f"--name={IDENTITY}", "42"),
        ),
    ],
)
def test_cancel_is_bound_to_the_allocation_name(job: JobRef, expected: tuple[str, ...]) -> None:
    transport = Scripted()
    scheduler(transport).cancel(ALLOCATION, job)
    assert transport.calls == [expected]


@pytest.mark.parametrize(
    "stderr",
    [
        b"scancel: error: Kill job error on job id 42: Job/step already completing or completed\n",
        b"scancel: error: Kill job error on job id 42: Invalid job id specified\n",
    ],
)
def test_cancel_of_an_ended_job_succeeds(stderr: bytes) -> None:
    scheduler(Scripted().queue("scancel", Completed(1, b"", stderr))).cancel(ALLOCATION, JobRef(42))


def test_cancel_failure_is_unavailable() -> None:
    transport = Scripted().queue("scancel", Completed(1, b"", b"scancel: error: Access denied\n"))
    with pytest.raises(Unavailable, match="scancel failed with exit status 1: scancel: error"):
        scheduler(transport).cancel(ALLOCATION, JobRef(42))
