"""The client's lease programs: writing an order and withdrawing it, scripted and run locally."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest
from scheduler_fixtures import ALLOCATION, LOG_ROOT, SLURM_BIN, Scripted, allocation, ok
from support.builders import target

from servatus.campaign._remote import Completed
from servatus.campaign._scheduler import ORDER, WITHDRAW, CommandFailed, Scheduler
from servatus.errors import Unavailable

LEASE = ALLOCATION
STEP = allocation(7)
ORDERS = f"{LOG_ROOT}/{LEASE}.d"


def scheduler(transport: Scripted) -> Scheduler:
    return Scheduler(transport, target(slurm_bin=SLURM_BIN, log_root=LOG_ROOT))


# --- Scheduler -------------------------------------------------------------------------------


def test_order_streams_the_order_into_the_lease_order_directory() -> None:
    transport = Scripted()
    scheduler(transport).order(LEASE, STEP, b"previous=0\n")
    assert transport.calls == [(*ORDER, ORDERS, STEP)]
    assert transport.stdins == [b"previous=0\n"]


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (
            Completed(1, b"", b"mv: cannot move\n"),
            f"ordering step {STEP} failed with exit status 1",
        ),
        (Completed(0, b"", b"sh: warning\n"), f"ordering step {STEP} failed with exit status 0"),
        (Unavailable("ssh: timeout"), "ssh: timeout"),
    ],
)
def test_a_failed_order_is_unavailable(reply: Completed | Unavailable, message: str) -> None:
    with pytest.raises(Unavailable, match=message):
        scheduler(Scripted().queue("sh", reply)).order(LEASE, STEP, b"")


@pytest.mark.parametrize(("stop", "stop_path"), [(True, f"{LOG_ROOT}/{STEP}.42.stop"), (False, "")])
def test_withdraw_names_the_claim_and_both_markers(stop: bool, stop_path: str) -> None:
    transport = Scripted().queue("sh", ok("withdrawn\n"))
    assert scheduler(transport).withdraw(LEASE, STEP, 42, stop=stop) is True
    assert transport.calls == [(*WITHDRAW, ORDERS, STEP, f"{LOG_ROOT}/{STEP}.42.void", stop_path)]
    assert transport.stdins == [b""]


def test_withdraw_reports_a_step_the_lease_took() -> None:
    transport = Scripted().queue("sh", ok("taken\n"))
    assert scheduler(transport).withdraw(LEASE, STEP, 42, stop=True) is False


@pytest.mark.parametrize(
    ("reply", "error", "message"),
    [
        (Completed(1, b"", b""), CommandFailed, "withdrawing step .* failed with exit status 1"),
        (
            Completed(0, b"withdrawn\n", b"ln: noise\n"),
            CommandFailed,
            "failed with exit status 0: ln: noise",
        ),
        (ok("withdrawn"), CommandFailed, "printed an unexpected reply"),
        (ok("taken\nwithdrawn\n"), CommandFailed, "printed an unexpected reply"),
        (ok(""), CommandFailed, "printed an unexpected reply"),
        (Unavailable("ssh: timeout"), Unavailable, "ssh: timeout"),
    ],
)
def test_a_failed_or_unclear_withdrawal_is_unavailable(
    reply: Completed | Unavailable, error: type[Unavailable], message: str
) -> None:
    with pytest.raises(error, match=message):
        scheduler(Scripted().queue("sh", reply)).withdraw(LEASE, STEP, 42, stop=False)


# --- The programs, run locally ---------------------------------------------------------------


def run(program: tuple[str, ...], *args: str | Path, stdin: bytes = b"") -> Completed:
    done = subprocess.run(
        (*program, *map(str, args)), input=stdin, capture_output=True, check=False, timeout=30
    )
    return Completed(done.returncode, done.stdout, done.stderr)


def mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def test_order_creates_a_private_directory_and_renames_the_order_into_place(
    tmp_path: Path,
) -> None:
    orders = tmp_path / f"{LEASE}.d"
    assert run(ORDER, orders, STEP, stdin=b"previous=7\n") == Completed(0, b"", b"")
    assert mode(orders) == 0o700 and orders.stat().st_uid == os.getuid()
    assert os.listdir(orders) == [f"{STEP}.run"]
    assert (orders / f"{STEP}.run").read_bytes() == b"previous=7\n"
    assert mode(orders / f"{STEP}.run") == 0o600
    assert run(ORDER, orders, allocation(8), stdin=b"x").returncode == 0
    assert sorted(os.listdir(orders)) == [f"{STEP}.run", f"{allocation(8)}.run"]


def test_order_refuses_an_order_directory_that_is_a_symbolic_link(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    (tmp_path / f"{LEASE}.d").symlink_to(real)
    assert run(ORDER, tmp_path / f"{LEASE}.d", STEP, stdin=b"x").returncode == 1
    assert os.listdir(real) == []


def withdraw(tmp_path: Path, *, stop: bool = True) -> Completed:
    orders = tmp_path / f"{LEASE}.d"
    stop_path = tmp_path / f"{STEP}.42.stop" if stop else ""
    return run(WITHDRAW, orders, STEP, tmp_path / f"{STEP}.42.void", stop_path)


def claim(tmp_path: Path) -> str:
    return os.readlink(tmp_path / f"{LEASE}.d" / f"{STEP}.claim")


def test_withdraw_wins_an_untaken_order_once_and_for_good(tmp_path: Path) -> None:
    assert withdraw(tmp_path) == Completed(0, b"withdrawn\n", b"")
    assert claim(tmp_path) == "withdrawn" and mode(tmp_path / f"{LEASE}.d") == 0o700
    assert (tmp_path / f"{STEP}.42.void").exists() and (tmp_path / f"{STEP}.42.stop").exists()
    assert withdraw(tmp_path, stop=False) == Completed(0, b"withdrawn\n", b"")


def test_withdraw_loses_an_order_the_lease_took(tmp_path: Path) -> None:
    orders = tmp_path / f"{LEASE}.d"
    orders.mkdir(mode=0o700)
    (orders / f"{STEP}.claim").symlink_to("taken")
    assert withdraw(tmp_path) == Completed(0, b"taken\n", b"")
    assert claim(tmp_path) == "taken"
    assert (tmp_path / f"{STEP}.42.stop").exists()
    assert not (tmp_path / f"{STEP}.42.void").exists()


def test_withdraw_without_stop_writes_no_stop_marker(tmp_path: Path) -> None:
    assert withdraw(tmp_path, stop=False) == Completed(0, b"withdrawn\n", b"")
    assert set(os.listdir(tmp_path)) == {f"{LEASE}.d", f"{STEP}.42.void"}


def test_withdraw_writes_the_stop_marker_before_it_claims(tmp_path: Path) -> None:
    orders = tmp_path / f"{LEASE}.d"
    missing = tmp_path / "missing" / f"{STEP}.42.stop"
    done = run(WITHDRAW, orders, STEP, tmp_path / f"{STEP}.42.void", missing)
    assert done.returncode == 1 and done.stdout == b""
    assert os.listdir(orders) == [] and not (tmp_path / f"{STEP}.42.void").exists()


def test_withdraw_keeps_an_earlier_stop_request(tmp_path: Path) -> None:
    stop_path = tmp_path / f"{STEP}.42.stop"
    stop_path.touch()
    os.utime(stop_path, (1_000_000, 1_000_000))
    assert withdraw(tmp_path).stdout == b"withdrawn\n"
    assert stop_path.stat().st_mtime == 1_000_000


def test_withdraw_refuses_an_order_directory_that_is_a_symbolic_link(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    (tmp_path / f"{LEASE}.d").symlink_to(real)
    assert withdraw(tmp_path).returncode == 1
    assert os.listdir(real) == [] and set(os.listdir(tmp_path)) == {f"{LEASE}.d", "real"}
