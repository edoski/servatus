"""Run a rendered lease and its orders under real local shells with a stub ``srun``.

The client's side is real too: the ``ORDER`` and ``WITHDRAW`` programs run locally to write each
order and contend for its claim. The ``srun`` stub records its options, waits for room while its
step's ``.wait`` file exists (recording a SIGUSR1 and dying of SIGTERM meanwhile, as a pending
``srun`` is killed), then becomes the step: it executes the start wrapper, which executes the
worker. So ``$!`` in an order is the worker's pid, and the SIGUSR1 and SIGTERM the lease sends its
``srun`` reach the worker as Slurm's ``srun`` forwards them. Nothing contacts Slurm. Only the poll
interval is patched.

The interrupt trap signals its whole process group, so every lease runs in its own session.
"""

from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from support.builders import target

from servatus.campaign._config import Resources, Task
from servatus.campaign._scheduler import ORDER, WITHDRAW
from servatus.campaign._script import POLL_SECONDS, Chained, render_lease, render_order

LEASE = "0123456789abcdef01234567"
JOB = "4242"
END_TIME = "1798761600"
SLURM = {"SLURM_JOB_ID": JOB, "SLURM_JOB_END_TIME": END_TIME}
SHELLS = [
    shell for shell in ("/bin/sh", "/bin/dash", "/bin/bash", "/bin/ksh") if Path(shell).exists()
]
# The stub interpreter itself injects these on macOS (PEP 538 locale coercion, CoreFoundation).
INTERPRETER_NOISE = ("__CF_USER_TEXT_ENCODING", "LC_CTYPE")
CLIENT_ENV = {"PATH": "/usr/bin:/bin"}
BUDGET = timedelta(hours=30)
SPENT = timedelta(seconds=5400)

SRUN = """#!{python}
import json, os, signal, sys, time
args = sys.argv[1:]
options = []
while args and args[0].startswith("--"):
    options.append(args.pop(0))
name = next(item.split("=", 1)[1] for item in options if item.startswith("--job-name="))
record = os.path.join({records!r}, name)
with open(record + ".part", "w") as handle:
    json.dump(options, handle)
os.rename(record + ".part", record + ".json")
def pending(*_):
    open(record + ".usr1", "w").close()
signal.signal(signal.SIGUSR1, pending)
while os.path.exists(record + ".wait"):
    time.sleep(0.02)
signal.signal(signal.SIGUSR1, signal.SIG_DFL)
if any(item.startswith("--gres=") for item in options):
    os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-step"
os.execv(args[0], args)
"""

# Each signal the worker handles is appended to a file with its arrival time.
WORKER = """#!{python}
import json, os, signal, sys, time
out, mode = sys.argv[1], sys.argv[2]
def note(suffix):
    with open(out + suffix, "a") as handle:
        handle.write(repr(time.time()) + "\\n")
def usr1(*_):
    note(".usr1")
    if mode.startswith("usr1="):
        os._exit(int(mode[5:]))
def term(*_):
    note(".term")
    os._exit(143)
signal.signal(signal.SIGUSR1, usr1)
signal.signal(signal.SIGTERM, term)
data = sys.stdin.buffer.read()
environment = {{k: v for k, v in os.environ.items() if k not in {noise!r}}}
with open(out + ".json", "w") as handle:
    json.dump({{"stdin": data.hex(), "env": environment}}, handle)
open(out + ".started", "w").close()
if mode.startswith("exit="):
    sys.exit(int(mode[5:]))
while True:
    time.sleep(0.05)
"""


def wait_until(condition: Callable[[], bool], what: str, seconds: float = 20) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, what
        time.sleep(0.02)


def step(index: int) -> str:
    return f"{index + 1:024x}"


def link(budget: timedelta | None = None, spent: timedelta = timedelta(0)) -> Chained:
    return Chained(1, budget, (1,), (spent,))


class Lease:
    def __init__(self, root: Path, *, grace: int = 1, gpus: int = 0) -> None:
        self.root = root
        self.bin = root / "bin"
        self.out = root / "out"
        self.records = root / "records"
        self.logs = root / "run logs"
        for directory in (self.bin, self.out, self.records, self.logs, root / "work"):
            directory.mkdir()
        python = sys.executable
        self._stub("srun", SRUN.format(python=python, records=str(self.records)))
        self.worker = self._stub("worker", WORKER.format(python=python, noise=INTERPRETER_NOISE))
        self.target = target(
            slurm_bin=str(self.bin),
            work_root=str(root / "work"),
            log_root=str(self.logs),
            container=None,
            host=None,
            max_script_bytes=16 * 1024 * 1024,
        )
        self.resources = Resources(
            cpus=2,
            memory_mib=100,
            time_limit="00:10:00",
            gpus=gpus,
            signal_before_end=timedelta(seconds=grace),
        )
        self.orders = self.logs / f"{LEASE}.d"
        poll = f"sleep {POLL_SECONDS}".encode()
        script = render_lease(self.target, self.resources, LEASE)
        assert script.count(poll) == 2
        self.script = root / "lease.sh"
        self.script.write_bytes(script.replace(poll, b"sleep 0.1"))
        self.errors = root / "lease.err"

    def _stub(self, name: str, text: str) -> Path:
        path = self.bin / name
        path.write_text(text)
        path.chmod(0o700)
        return path

    def task(self, key: str, mode: str, stdin: bytes = b"", **env: str) -> Task:
        return Task(key, (str(self.worker), str(self.out / key), mode), stdin=stdin, env=env)

    def order(self, step_id: str, task: Task, chain: Chained | None = None) -> None:
        script = render_order(self.target, self.resources, task, step_id, chain or link())
        result = subprocess.run(
            [*ORDER, str(self.orders), step_id],
            input=script,
            env=CLIENT_ENV,
            capture_output=True,
            timeout=20,
            check=False,
        )
        assert (result.returncode, result.stderr) == (0, b"")

    def withdrawal(self, step_id: str, *, stop: bool) -> list[str]:
        stop_path = str(self.marker(step_id, "stop")) if stop else ""
        return [*WITHDRAW, str(self.orders), step_id, str(self.marker(step_id, "void")), stop_path]

    def withdraw(self, step_id: str, *, stop: bool) -> str:
        result = subprocess.run(
            self.withdrawal(step_id, stop=stop),
            env=CLIENT_ENV,
            capture_output=True,
            timeout=20,
            check=False,
        )
        assert (result.returncode, result.stderr) == (0, b"")
        return result.stdout.decode()

    def marker(self, name: str, suffix: str) -> Path:
        return self.logs / f"{name}.{JOB}.{suffix}"

    def markers(self, name: str) -> set[str]:
        prefix = f"{name}.{JOB}."
        return {
            path.name.removeprefix(prefix)
            for path in self.logs.iterdir()
            if path.name.startswith(prefix)
        }

    def end(self, step_id: str) -> int | None:
        ends = [name for name in self.markers(step_id) if name.endswith(".end")]
        assert len(ends) <= 1, ends
        return int(ends[0].split(".")[1]) if ends else None

    def claim(self, step_id: str) -> str | None:
        path = self.orders / f"{step_id}.claim"
        return os.readlink(path) if path.is_symlink() else None

    def stderr(self) -> str:
        # bash, unlike dash or ksh, reports a subshell that the relayed SIGUSR1 reached between fork
        # and its own trap (macOS /bin/sh is bash 3.2). The subshell dies as it would in any shell.
        return "".join(
            line
            for line in self.errors.read_text().splitlines(keepends=True)
            if "run_pending_traps:" not in line
        )

    def started(self, key: str) -> bool:
        return (self.out / f"{key}.started").exists()

    def record(self, key: str) -> dict[str, Any]:
        return json.loads((self.out / f"{key}.json").read_text())

    def signals(self, key: str, kind: str = "usr1") -> list[float]:
        path = self.out / f"{key}.{kind}"
        return [float(line) for line in path.read_text().split()] if path.exists() else []

    def srun_options(self, step_id: str) -> list[str] | None:
        path = self.records / f"servatus-{step_id}-0.json"
        return json.loads(path.read_text()) if path.exists() else None

    def wait_for_room(self, step_id: str) -> None:
        (self.records / f"servatus-{step_id}-0.wait").touch()

    @contextmanager
    def spawn(self, shell: str) -> Generator[subprocess.Popen[bytes]]:
        with self.errors.open("wb") as errors:
            child = subprocess.Popen(
                [shell, str(self.script)],
                env=SLURM,
                cwd=self.root,
                stdout=subprocess.DEVNULL,
                stderr=errors,
                start_new_session=True,
            )
        try:
            yield child
        finally:
            # Reap anything the test left behind, steps included.
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=20)


def mtime(path: Path) -> int:
    return path.stat().st_mtime_ns


@pytest.mark.parametrize("shell", SHELLS)
def test_a_lease_runs_each_order_it_claims_with_exact_stdin_and_environment(
    tmp_path: Path, shell: str
) -> None:
    lease = Lease(tmp_path)
    ignored = (
        "0123456789ABCDEF01234567",
        "0123",
        "0123456789abcdef012345678",
        "0123456789abcdef0123456g",
    )
    for index, name in enumerate(ignored):
        lease.order(name, lease.task(f"ignored{index}", "exit=0"))
    stdin = bytes(range(256)) * 4 + b"%s%d\\c'\"\n\\0101"
    first, second = step(0), step(1)
    task = lease.task("first", "exit=7", stdin, OMP_NUM_THREADS="4")
    lease.order(first, task, link(BUDGET, SPENT))
    with lease.spawn(shell) as child:
        wait_until(lambda: lease.end(first) is not None, "the first step never ended")
        lease.order(second, lease.task("second", "exit=0"))
        wait_until(lambda: lease.end(second) is not None, "a later order was never taken")
        assert child.poll() is None
    assert (lease.claim(first), lease.claim(second)) == ("taken", "taken")
    assert lease.markers(first) == {"start", "0.7.end"}
    assert lease.markers(second) == {"start", "0.0.end"}
    record = lease.record("first")
    assert bytes.fromhex(record["stdin"]) == stdin
    assert record["env"] == {
        "OMP_NUM_THREADS": "4",
        "SERVATUS_TASK_KEY": "first",
        "SERVATUS_ALLOCATION_ID": first,
        "SERVATUS_SLOT": "0",
        "SERVATUS_LINK": "1",
        "SERVATUS_LINKS": "1",
        "SERVATUS_BUDGET": "108000",
        "SERVATUS_JOB_ID": JOB,
        "SERVATUS_JOB_END_TIME": END_TIME,
        "SERVATUS_RUN_TIME": "5400",
    }
    environment = lease.record("second")["env"]
    assert "SERVATUS_BUDGET" not in environment
    assert environment["SERVATUS_RUN_TIME"] == "0"
    claims = {path.name for path in lease.orders.iterdir() if path.name.endswith(".claim")}
    assert claims == {f"{first}.claim", f"{second}.claim"}
    assert not any(lease.started(f"ignored{index}") for index in range(len(ignored)))
    assert lease.markers(LEASE) == {"start"}
    assert lease.stderr() == ""


@pytest.mark.parametrize("shell", SHELLS)
def test_an_order_withdrawn_before_the_lease_takes_it_never_runs(
    tmp_path: Path, shell: str
) -> None:
    lease = Lease(tmp_path)
    gone, kept = step(0), step(1)
    lease.order(gone, lease.task("gone", "exit=0"))
    assert lease.withdraw(gone, stop=False) == "withdrawn\n"
    assert lease.withdraw(gone, stop=False) == "withdrawn\n"
    lease.order(kept, lease.task("kept", "exit=0"))
    with lease.spawn(shell):
        # Both orders are listed by the same poll, the withdrawn one first.
        wait_until(lambda: lease.end(kept) is not None, "the kept step never ended")
        assert lease.withdraw(kept, stop=True) == "taken\n"
    assert lease.claim(gone) == "withdrawn"
    assert lease.markers(gone) == {"void"}
    assert not lease.started("gone")
    assert lease.srun_options(gone) is None
    assert lease.claim(kept) == "taken"
    assert lease.markers(kept) == {"start", "0.0.end", "stop"}
    assert lease.stderr() == ""


ROUNDS = 15
ORDERS_PER_ROUND = 6


@pytest.mark.parametrize("shell", SHELLS)
def test_a_lease_and_a_withdrawal_racing_for_an_order_agree_on_one_winner(
    tmp_path: Path, shell: str
) -> None:
    lease = Lease(tmp_path)
    pace = random.Random(shell)
    answers: dict[str, str] = {}
    with lease.spawn(shell) as child:
        for round_ in range(ROUNDS):
            steps = [step(round_ * ORDERS_PER_ROUND + index) for index in range(ORDERS_PER_ROUND)]
            # Odd rounds start withdrawing first, so a slow host still sees both sides win.
            delay = pace.uniform(0, 0.15)
            if round_ % 2 == 0:
                for step_id in steps:
                    lease.order(step_id, lease.task(step_id, "exit=0"))
                time.sleep(delay)
            withdrawals = [
                (
                    step_id,
                    subprocess.Popen(
                        lease.withdrawal(step_id, stop=False),
                        env=CLIENT_ENV,
                        stdout=subprocess.PIPE,
                    ),
                )
                for step_id in steps
            ]
            if round_ % 2 == 1:
                time.sleep(delay)
                for step_id in steps:
                    lease.order(step_id, lease.task(step_id, "exit=0"))
            for step_id, withdrawal in withdrawals:
                answers[step_id] = withdrawal.communicate(timeout=20)[0].decode().strip()
                assert withdrawal.returncode == 0
        taken = [step_id for step_id, answer in answers.items() if answer == "taken"]
        wait_until(lambda: all(lease.end(s) is not None for s in taken), "a taken step hung")
        time.sleep(0.3)
        assert child.poll() is None
    for step_id, answer in answers.items():
        assert lease.claim(step_id) == answer, step_id
        expected = {"start", "0.0.end"} if answer == "taken" else {"void"}
        assert lease.markers(step_id) == expected, step_id
    assert set(answers.values()) == {"taken", "withdrawn"}, "the race was never contended"


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize("mode", ["usr1=75", "hold"], ids=["yields", "ignores-usr1"])
def test_a_stop_signals_a_running_step_then_terminates_it_after_grace(
    tmp_path: Path, shell: str, mode: str
) -> None:
    lease = Lease(tmp_path, grace=1)
    step_id = step(0)
    lease.order(step_id, lease.task("t", mode))
    with lease.spawn(shell):
        wait_until(lambda: lease.started("t"), "the step never started")
        assert lease.withdraw(step_id, stop=True) == "taken\n"
        wait_until(lambda: lease.end(step_id) is not None, "the stopped step never ended")
    usr1 = lease.signals("t")
    assert len(usr1) == 1
    if mode == "hold":
        assert lease.end(step_id) == 143
        (term,) = lease.signals("t", "term")
        assert term - usr1[0] >= 0.9, "SIGTERM came before the grace ran out"
    else:
        assert lease.end(step_id) == 75
    code = lease.end(step_id)
    assert lease.markers(step_id) == {"start", "stop", f"0.{code}.end"}
    assert mtime(lease.marker(step_id, "stop")) < mtime(lease.marker(step_id, f"0.{code}.end"))
    assert lease.stderr() == ""


@pytest.mark.parametrize("shell", SHELLS)
def test_a_stop_terminates_a_step_still_waiting_for_room_at_once(
    tmp_path: Path, shell: str
) -> None:
    lease = Lease(tmp_path, grace=30)
    step_id = step(0)
    lease.wait_for_room(step_id)
    lease.order(step_id, lease.task("t", "hold"))
    with lease.spawn(shell):
        wait_until(lambda: lease.srun_options(step_id) is not None, "the step was never launched")
        assert lease.withdraw(step_id, stop=True) == "taken\n"
        wait_until(
            lambda: lease.end(step_id) is not None, "a waiting step outlived its stop", seconds=10
        )
    code = lease.end(step_id)
    assert code not in (None, 0)
    assert not (lease.records / f"servatus-{step_id}-0.usr1").exists()
    assert not lease.started("t")
    assert lease.markers(step_id) == {"stop", f"0.{code}.end"}


@pytest.mark.parametrize("shell", SHELLS)
def test_the_pre_limit_signal_is_relayed_once_and_stops_still_work_after_it(
    tmp_path: Path, shell: str
) -> None:
    lease = Lease(tmp_path, grace=1)
    early, late, after = step(0), step(1), step(2)
    lease.order(early, lease.task("early", "hold"))
    lease.order(late, lease.task("late", "hold"))
    with lease.spawn(shell) as child:
        wait_until(lambda: lease.started("early") and lease.started("late"), "steps never started")
        assert lease.withdraw(early, stop=True) == "taken\n"
        wait_until(lambda: len(lease.signals("early")) == 1, "the stop sent no SIGUSR1")
        child.send_signal(signal.SIGUSR1)
        wait_until(lambda: lease.marker(LEASE, "limit").exists(), "no limit marker")
        wait_until(lambda: len(lease.signals("late")) == 1, "the signal was not relayed")
        wait_until(lambda: lease.end(early) is not None, "the early stop never ended its step")
        lease.order(after, lease.task("after", "exit=0"))
        assert lease.withdraw(late, stop=True) == "taken\n"
        wait_until(lambda: lease.end(late) is not None, "a stop after the signal was ignored")
        assert child.poll() is None
    early_usr1 = lease.signals("early")
    assert len(early_usr1) == 2, "the stopped step got the relayed signal other than once"
    (early_term,) = lease.signals("early", "term")
    assert early_term - early_usr1[0] >= 0.9, "the relayed signal cut the grace short"
    late_usr1 = lease.signals("late")
    assert len(late_usr1) == 2
    (late_term,) = lease.signals("late", "term")
    assert late_term - late_usr1[1] >= 0.9
    assert (lease.end(early), lease.end(late)) == (143, 143)
    assert lease.claim(after) is None
    assert not lease.started("after")
    assert lease.markers(LEASE) == {"start", "limit"}
    assert lease.stderr() == ""


@pytest.mark.parametrize("shell", SHELLS)
def test_an_interrupt_records_the_stop_then_terminates_every_step(
    tmp_path: Path, shell: str
) -> None:
    lease = Lease(tmp_path)
    steps = {"a": step(0), "b": step(1)}
    for key, step_id in steps.items():
        lease.order(step_id, lease.task(key, "hold"))
    with lease.spawn(shell) as child:
        wait_until(lambda: all(map(lease.started, steps)), "steps never started")
        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=20) == 1
        for key in steps:
            wait_until(lambda key=key: bool(lease.signals(key, "term")), f"{key} kept running")
        time.sleep(0.3)
    assert lease.markers(LEASE) == {"start", "stop"}
    for step_id in steps.values():
        assert lease.markers(step_id) == {"start"}, "an interrupted step recorded an outcome"


@pytest.mark.parametrize("variable", sorted(SLURM))
@pytest.mark.parametrize("shell", SHELLS)
def test_a_lease_without_a_job_id_fails_before_any_marker(
    tmp_path: Path, shell: str, variable: str
) -> None:
    lease = Lease(tmp_path)
    result = subprocess.run(
        [shell, str(lease.script)],
        env={name: value for name, value in SLURM.items() if name != variable},
        cwd=tmp_path,
        capture_output=True,
        timeout=20,
        check=False,
        start_new_session=True,
    )
    assert result.returncode != 0
    assert f"Slurm did not set {variable}".encode() in result.stderr
    assert list(lease.logs.iterdir()) == []


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize("gpus", [0, 1])
def test_a_step_asks_srun_for_one_slot(tmp_path: Path, shell: str, gpus: int) -> None:
    lease = Lease(tmp_path, gpus=gpus)
    step_id = step(0)
    lease.order(step_id, lease.task("t", "exit=0"))
    with lease.spawn(shell):
        wait_until(lambda: lease.end(step_id) is not None, "the step never ended")
    assert lease.end(step_id) == 0
    options = lease.srun_options(step_id)
    assert options is not None
    assert "--cpus-per-task=2" in options
    assert "--mem=100M" in options
    gres = [option for option in options if option.startswith("--gres=")]
    assert gres == (["--gres=gpu:a100:1"] if gpus else [])
