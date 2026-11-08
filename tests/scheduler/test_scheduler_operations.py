from __future__ import annotations

from pathlib import PurePosixPath

import pytest
from scheduler_fixtures import (
    ALLOCATION,
    AT,
    IDENTITY,
    LOG_ROOT,
    SLURM_BIN,
    Scripted,
    allocation,
    listing,
    ok,
)
from support.builders import target

from servatus.campaign._evidence import End, QueueRow, Start, Stop
from servatus.campaign._remote import Completed
from servatus.campaign._scheduler import WRITABLE_DIRECTORY, CommandFailed, Scheduler
from servatus.errors import ConfigurationError, EvidenceConflict, Unavailable

SBATCH = f"{SLURM_BIN}/sbatch"
ARGV = (SBATCH, "--parsable", "--no-requeue", "--export=NIL", f"--job-name={IDENTITY}")
ROOT = PurePosixPath(LOG_ROOT)
PRINTF = ("-printf", "%T@ %p\n")
FIND = (
    "/usr/bin/find",
    "-H",
    LOG_ROOT,
    "-maxdepth",
    "1",
    "-type",
    "f",
    "(",
    "-name",
    "*.start",
    "-o",
    "-name",
    "*.stop",
    "-o",
    "-name",
    "*.limit",
    "-o",
    "-name",
    "*.end",
    "-o",
    "-name",
    "*.go",
    ")",
    *PRINTF,
)


def scheduler(transport: Scripted) -> Scheduler:
    return Scheduler(transport, target(slurm_bin=SLURM_BIN, log_root=LOG_ROOT))


def squeue_argv(*allocation_ids: str) -> tuple[str, ...]:
    names = ",".join(f"servatus-{allocation_id}" for allocation_id in allocation_ids)
    return (
        f"{SLURM_BIN}/squeue",
        "--me",
        "--all",
        "--states=all",
        "--noheader",
        f"--name={names}",
        "--format=%i|%j|%T",
    )


# --- queue -----------------------------------------------------------------------------------


def test_queue_issues_the_exact_query_and_groups_rows_by_allocation() -> None:
    other = allocation(9)
    transport = Scripted().queue(
        "squeue", ok(f"42|{IDENTITY}|COMPLETED\n43|servatus-{other}|PENDING\n")
    )
    rows = scheduler(transport).queue([ALLOCATION, other, allocation(8)])
    assert transport.calls == [squeue_argv(ALLOCATION, other, allocation(8))]
    assert rows == {
        ALLOCATION: (QueueRow(42, "COMPLETED"),),
        other: (QueueRow(43, "PENDING"),),
        allocation(8): (),
    }


def test_queue_batches_at_most_one_hundred_names_per_query() -> None:
    ids = [allocation(index) for index in range(250)]
    transport = Scripted()
    rows = scheduler(transport).queue(ids)
    assert transport.calls == [
        squeue_argv(*ids[:100]),
        squeue_argv(*ids[100:200]),
        squeue_argv(*ids[200:]),
    ]
    assert rows == {allocation_id: () for allocation_id in ids}


def test_queue_of_nothing_issues_no_commands() -> None:
    transport = Scripted()
    assert scheduler(transport).queue([]) == {}
    assert transport.calls == []


@pytest.mark.parametrize(
    "failure",
    [
        Completed(1, b"", b"squeue: error: communication failure\n"),
        Completed(1, f"42|{IDENTITY}|RUNNING\n".encode(), b""),
        Completed(0, f"42|{IDENTITY}|RUNNING\n".encode(), b"squeue: warning\n"),
    ],
)
def test_queue_failure_or_stderr_is_unavailable(failure: Completed) -> None:
    with pytest.raises(Unavailable, match="squeue failed with exit status"):
        scheduler(Scripted().queue("squeue", failure)).queue([ALLOCATION])


def test_queue_transport_failure_propagates() -> None:
    transport = Scripted().queue("squeue", Unavailable("ssh: connection refused"))
    with pytest.raises(Unavailable, match="connection refused"):
        scheduler(transport).queue([ALLOCATION])


def test_queue_row_of_an_unrelated_job_is_a_conflict() -> None:
    transport = Scripted().queue("squeue", ok("42|servatus-other|RUNNING\n"))
    with pytest.raises(EvidenceConflict, match="unrelated job"):
        scheduler(transport).queue([ALLOCATION])


# --- markers ---------------------------------------------------------------------------------


def test_markers_of_a_few_allocations_are_named_exactly_in_one_read() -> None:
    other = allocation(9)
    reply = listing(
        f"{ALLOCATION}.42.start",
        f"{ALLOCATION}.42.1.3.end",
        f"{other}.43.stop",
        f"{allocation(7)}.44.start",
    )
    transport = Scripted().queue("find", ok(reply))
    found = scheduler(transport).markers({ALLOCATION: 2, other: 1})
    names = tuple(sorted((f"{ALLOCATION}.*", f"{other}.*")))
    assert transport.calls == [
        (*FIND[:8], "-name", names[0], "-o", "-name", names[1], ")", *PRINTF)
    ]
    assert found == {ALLOCATION: (Start(42, AT), End(42, 1, 3, AT)), other: (Stop(43, AT),)}


def test_exact_names_stay_within_the_argument_bound() -> None:
    ids = {allocation(index): 1 for index in range(7)}
    transport = Scripted()
    scheduler(transport).markers(ids)
    assert len(transport.calls) == 1 and len(transport.calls[0]) == 31
    assert transport.calls[0][9:-3:3] == tuple(sorted(f"{name}.*" for name in ids))


def test_markers_split_the_listing_by_leading_hex_digit_only_when_large() -> None:
    first, second, third = f"a{ALLOCATION[1:]}", f"3{ALLOCATION[1:]}", f"a{allocation(1)[1:]}"
    many = {allocation(index): 4 for index in range(10, 18)}
    small = Scripted()
    scheduler(small).markers(many)
    assert small.calls == [FIND]

    large = Scripted().queue(
        "find",
        ok(listing(f"{second}.30.start")),
        ok(listing(f"{first}.10.start", f"{third}.11.stop")),
    )
    extra = {f"a{allocation(index)[1:]}": 1 for index in range(20, 26)}
    found = scheduler(large).markers({first: 20_000, second: 20_000, third: 1} | extra)
    assert [call[9:-3:3] for call in large.calls] == [
        ("3*.start", "3*.stop", "3*.limit", "3*.end", "3*.go"),
        ("a*.start", "a*.stop", "a*.limit", "a*.end", "a*.go"),
    ]
    assert all(call[:8] == FIND[:8] and call[-3:] == (")", *PRINTF) for call in large.calls)
    assert {name: found[name] for name in (first, second, third)} == {
        first: (Start(10, AT),),
        second: (Start(30, AT),),
        third: (Stop(11, AT),),
    }


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (
            Completed(1, b"", f"find: {LOG_ROOT}: No such file or directory\n".encode()),
            f"listing {LOG_ROOT} failed with exit status 1: find: ",
        ),
        (Completed(1, listing(f"{ALLOCATION}.42.start"), b""), "failed with exit status 1"),
        (
            Completed(0, listing(f"{ALLOCATION}.42.start"), b"find: permission denied\n"),
            "failed with exit status 0: find: permission denied",
        ),
    ],
)
def test_marker_listing_failure_or_stderr_is_unavailable(failure: Completed, message: str) -> None:
    with pytest.raises(Unavailable, match=message):
        scheduler(Scripted().queue("find", failure)).markers({ALLOCATION: 1})


def test_malformed_marker_listing_is_a_conflict() -> None:
    transport = Scripted().queue("find", ok(listing(f"{ALLOCATION}.42.0.256.end")))
    with pytest.raises(EvidenceConflict, match=f"allocation {ALLOCATION} has a malformed"):
        scheduler(transport).markers({ALLOCATION: 1})


# --- submit, test_only, ping -----------------------------------------------------------------


def test_submit_streams_the_exact_script_and_parses_the_receipt() -> None:
    transport = Scripted().queue("sbatch", ok("42\n"))
    script = b"#!/bin/sh\n\x00\xff"
    assert scheduler(transport).submit(ARGV, script) == 42
    assert transport.calls == [ARGV]
    assert transport.stdins == [script]


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (
            Completed(1, b"", b"sbatch: error: Invalid account\n"),
            "status 1: sbatch: error: Invalid",
        ),
        (Completed(0, b"42;alpha\n", b""), "exactly one job number"),
        (Completed(0, b"Submitted batch job 42\n", b""), "exactly one job number"),
        (Completed(0, b"", b""), "exactly one job number"),
        (Unavailable("deadline"), "deadline"),
    ],
)
def test_submit_failures_are_unavailable(reply: Completed | Unavailable, message: str) -> None:
    with pytest.raises(Unavailable, match=message):
        scheduler(Scripted().queue("sbatch", reply)).submit(ARGV, b"#!/bin/sh\n")


def test_command_bounds_are_configuration_errors_before_contact() -> None:
    transport = Scripted()
    with pytest.raises(ConfigurationError, match="bound of 4096 bytes"):
        scheduler(transport).submit((*ARGV, "--chdir=" + "x" * 4096), b"")
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
    with pytest.raises(Unavailable, match="log is unavailable") as caught:
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


# --- cancel ----------------------------------------------------------------------------------


def test_cancel_names_only_this_users_jobs_with_the_allocation_identity() -> None:
    transport = Scripted()
    scheduler(transport).cancel(ALLOCATION)
    assert transport.calls == [(f"{SLURM_BIN}/scancel", "--me", f"--name={IDENTITY}")]


def test_cancel_tolerates_stderr_when_it_exits_zero() -> None:
    transport = Scripted().queue("scancel", Completed(0, b"", b"scancel: notice\n"))
    scheduler(transport).cancel(ALLOCATION)


@pytest.mark.parametrize(
    "failure",
    [Completed(1, b"", b"scancel: error: Access denied\n"), Unavailable("ssh: timeout")],
)
def test_cancel_failure_is_unavailable(failure: Completed | Unavailable) -> None:
    transport = Scripted().queue("scancel", failure)
    with pytest.raises(Unavailable, match="scancel failed with exit status 1: scancel: error|ssh"):
        scheduler(transport).cancel(ALLOCATION)


# --- doctor facts ----------------------------------------------------------------------------


def test_config_reads_each_setting_as_slurm_prints_it() -> None:
    reply = ok(
        "Configuration data as of 2026-11-06T17:00:00\n"
        "MinJobAge               = 300 sec\n"
        "SlurmctldParameters     = a=b,c\n"
        "\n"
        "Slurmctld(primary) at giano is UP\n"
    )
    transport = Scripted().queue("scontrol", reply)
    assert scheduler(transport).config() == {
        "MinJobAge": "300 sec",
        "SlurmctldParameters": "a=b,c",
    }
    assert transport.calls == [(f"{SLURM_BIN}/scontrol", "show", "config")]


def test_config_failure_is_unavailable() -> None:
    transport = Scripted().queue("scontrol", Completed(1, b"", b"scontrol: error: no\n"))
    with pytest.raises(Unavailable, match="scontrol show config failed .* no"):
        scheduler(transport).config()


def test_writable_directory_asks_a_shell_and_reads_its_status() -> None:
    replies = (ok(""), Completed(1, b"", b""), Completed(2, b"", b"x\n"), Completed(-9, b"", b""))
    transport = Scripted().queue("sh", *replies)
    path = PurePosixPath("/cluster/logs")
    assert scheduler(transport).writable_directory(path) is True
    assert transport.calls[0] == (*WRITABLE_DIRECTORY, str(path))
    assert scheduler(transport).writable_directory(path) is False
    with pytest.raises(CommandFailed, match="checking /cluster/logs failed with exit status 2: x"):
        scheduler(transport).writable_directory(path)
    with pytest.raises(CommandFailed, match="exit status -9"):
        scheduler(transport).writable_directory(path)
