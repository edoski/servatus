from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import pytest
from scheduler_fixtures import ALLOCATION, IDENTITY, query, sacct_row
from support.builders import target

import servatus.campaign._remote as _remote
from servatus.campaign._evidence import AllocationState, JobRef
from servatus.campaign._remote import Completed, Local, Ssh, check_command, connect
from servatus.campaign._scheduler import Scheduler
from servatus.errors import ConfigurationError, Unavailable

FAKE_SSH = """#!{python}
import json, os, sys, time
config = json.load(open({config!r}))
args = sys.argv[1:]
with open(config["record"], "w") as handle:
    json.dump({{"argv": args, "env": dict(os.environ)}}, handle)
mode = config["mode"]
if mode == "refused":
    sys.stderr.write("ssh: connect to host login port 22: Connection refused\\n")
    sys.exit(255)
if mode == "unmarked":
    sys.stdout.write("42\\n")
    sys.exit(0)
if mode == "sleep":
    time.sleep(60)
sys.stderr.write("Authorized users only. Activity is monitored.\\n")
sys.stderr.flush()
sys.stdout.write("Loading conda environment... done")
sys.stdout.flush()
if mode == "quiet-stderr":
    fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(fd, 2)
os.execv("/bin/sh", ["/bin/sh", "-c", args[-1]])
"""


def fake_ssh(tmp_path: Path, mode: str = "banner") -> tuple[Ssh, Path]:
    record = tmp_path / "ssh-record.json"
    config = tmp_path / "ssh-config.json"
    config.write_text(json.dumps({"mode": mode, "record": str(record)}))
    executable = tmp_path / "fake-ssh"
    executable.write_text(FAKE_SSH.format(python=sys.executable, config=str(config)))
    executable.chmod(0o700)
    return Ssh("login.example.edu", command=(str(executable),)), record


def recorded(record: Path) -> dict[str, Any]:
    return json.loads(record.read_text())


# --- Bounds ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        ((), "nonempty sequence"),
        ("/bin/true", "sequence of arguments"),
        (("",), "program must be nonempty"),
        (("/bin/echo",) * 33, "bound of 32 arguments"),
        (("/bin/echo", "x" * 4097), "bound of 4096 bytes"),
        (("/bin/echo", *["x" * 4000] * 5), "bound of 16384 bytes"),
        (("/bin/echo", "a\0b"), "NUL"),
        (("/bin/echo", "\udcff"), "UTF-8"),
        (("/bin/echo", 3), "sequence of strings"),
    ],
)
def test_check_command_rejects_out_of_bound_commands(argv: Any, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        check_command(argv)


def test_check_command_accepts_the_largest_permitted_command() -> None:
    argv = ("/bin/echo", *["x" * 4095] * 3, "y" * (16 * 1024 - 3 * 4096 - 11))
    assert check_command(argv) == argv


def test_bound_violations_never_spawn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("spawned")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    ssh, _ = fake_ssh(tmp_path)
    for transport in (Local(), ssh):
        with pytest.raises(ConfigurationError, match="bound of 32 arguments"):
            transport.run(("/bin/echo",) * 33)
        with pytest.raises(ConfigurationError, match="max_stdout"):
            transport.run(("/bin/echo",), max_stdout=-1)


@pytest.mark.parametrize("host", ["", "-oProxyCommand=evil", "a b", "h" * 4097, "x\ny"])
def test_ssh_rejects_unsafe_destinations(host: str) -> None:
    with pytest.raises(ConfigurationError, match="SSH destination"):
        Ssh(host)


def test_connect_selects_ssh_or_local() -> None:
    assert connect(target(host=None)) == Local()
    remote = connect(target())
    assert isinstance(remote, Ssh)
    assert (remote.host, remote.command) == ("login.example.edu", ("ssh",))


# --- Local -----------------------------------------------------------------------------------


def test_local_runs_without_a_shell_under_a_scrubbed_utc_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TZ", "Pacific/Honolulu")
    monkeypatch.setenv("SBATCH_PARTITION", "hostile")
    result = Local().run(("/usr/bin/env",))
    assert result.returncode == 0
    assert sorted(result.stdout.decode().splitlines()) == [
        "LANG=C",
        "LC_ALL=C",
        "PATH=/usr/bin:/bin",
        "TZ=UTC",
    ]
    assert Local().run(("/bin/echo", "$HOME", "*")).stdout == b"$HOME *\n"


def test_local_streams_binary_stdin_and_drains_both_streams_concurrently() -> None:
    payload = bytes(range(256)) * 4096
    result = Local().run(
        ("/bin/sh", "-c", "cat; head -c 300000 /dev/zero >&2; exit 4"), stdin=payload
    )
    assert result == Completed(4, payload, b"\0" * 300000)


def test_unread_large_stdin_cannot_deadlock() -> None:
    started = time.monotonic()
    assert Local().run(("/usr/bin/true",), stdin=b"x" * 8_000_000).returncode == 0
    assert time.monotonic() - started < 10


def test_spawn_failure_is_unavailable() -> None:
    with pytest.raises(Unavailable, match="could not run: No such file"):
        Local().run(("/nonexistent/servatus-test-program",))


@pytest.mark.parametrize(
    ("argv", "max_stdout"),
    [
        (("/bin/sh", "-c", "head -c 2000 /dev/zero"), 1000),
        (("/bin/sh", "-c", "head -c 1048577 /dev/zero >&2"), 1024),
    ],
)
def test_output_beyond_its_bound_is_unavailable(argv: tuple[str, ...], max_stdout: int) -> None:
    with pytest.raises(Unavailable, match="byte bound"):
        Local().run(argv, max_stdout=max_stdout)


class Spawned:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.children: list[subprocess.Popen[bytes]] = []
        real = subprocess.Popen

        def spawn(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
            child = cast("subprocess.Popen[bytes]", real(*args, **kwargs))
            self.children.append(child)
            return child

        monkeypatch.setattr(subprocess, "Popen", spawn)

    def assert_reaped(self) -> None:
        (child,) = self.children
        assert child.returncode is not None
        assert child.stdout is not None and child.stdout.closed
        assert child.stderr is not None and child.stderr.closed


def wait_gone(pid: int) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    raise AssertionError(f"process {pid} survived")


def test_deadline_kills_and_reaps_the_whole_process_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spawned = Spawned(monkeypatch)
    monkeypatch.setattr(_remote, "DEADLINE_SECONDS", 0.5)
    pidfile = tmp_path / "grandchild"
    started = time.monotonic()
    with pytest.raises(Unavailable, match="deadline"):
        Local().run(("/bin/sh", "-c", f"/bin/sleep 30 & echo $! > {pidfile}; wait"))
    assert time.monotonic() - started < 5
    spawned.assert_reaped()
    wait_gone(int(pidfile.read_text()))


@pytest.mark.parametrize(
    ("failure", "expected", "message"),
    [
        (OSError("selector unavailable"), Unavailable, "could not run"),
        (KeyboardInterrupt("operator stop"), KeyboardInterrupt, "operator stop"),
    ],
)
def test_any_drain_failure_kills_and_reaps_the_child(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    expected: type[BaseException],
    message: str,
) -> None:
    spawned = Spawned(monkeypatch)

    def broken() -> selectors.BaseSelector:
        raise failure

    monkeypatch.setattr(selectors, "DefaultSelector", broken)
    with pytest.raises(expected, match=message):
        Local().run(("/bin/sleep", "30"), stdin=b"binary\0input")
    spawned.assert_reaped()


def test_overflow_kills_and_reaps_the_child(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned = Spawned(monkeypatch)
    with pytest.raises(Unavailable, match="byte bound"):
        Local().run(("/bin/sh", "-c", "head -c 5000 /dev/zero; exec /bin/sleep 30"), max_stdout=10)
    spawned.assert_reaped()


# --- Ssh -------------------------------------------------------------------------------------


def test_ssh_invocation_shape_and_remote_environment(tmp_path: Path) -> None:
    ssh, record = fake_ssh(tmp_path)
    result = ssh.run(("/usr/bin/env",))
    argv = recorded(record)["argv"]
    assert argv[:7] == [
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "LogLevel=ERROR",
        "--",
        "login.example.edu",
    ]
    assert len(argv) == 8 and argv[7].startswith("/bin/sh -c ")
    assert sorted(result.stdout.decode().splitlines()) == [
        "LANG=C",
        "LC_ALL=C",
        "PATH=/usr/bin:/bin",
        "TZ=UTC",
    ]


def test_ssh_discards_banner_and_shell_noise_before_the_markers(tmp_path: Path) -> None:
    # Regression: an sshd Banner on stderr or ~/.bashrc output on stdout used to corrupt every
    # scheduler reply (permanent observation failure, unparsable sbatch receipts).
    ssh, _ = fake_ssh(tmp_path)
    result = ssh.run(("/bin/sh", "-c", "printf 'out\\n'; printf 'err\\n' >&2; exit 3"))
    assert result == Completed(3, b"out\n", b"err\n")


def test_ssh_preserves_argv_and_binary_stdin_exactly(tmp_path: Path) -> None:
    ssh, _ = fake_ssh(tmp_path)
    marker = tmp_path / "not-executed"
    args = ("a b", "'quoted'", '"double"', f"$(touch {marker})", "", "λ", "x\ny", "*", "-n")
    result = ssh.run(("/bin/sh", "-c", 'printf "%s|" "$@"', "sh", *args))
    assert result.stdout.decode() == "".join(f"{item}|" for item in args)
    assert not marker.exists()
    payload = bytes(range(256)) * 4096
    assert ssh.run(("/bin/cat",), stdin=payload).stdout == payload


@pytest.mark.parametrize("mode", ["unmarked", "refused", "quiet-stderr"])
def test_ssh_without_both_markers_is_unavailable(tmp_path: Path, mode: str) -> None:
    ssh, _ = fake_ssh(tmp_path, mode)
    with pytest.raises(Unavailable, match="output marker missing"):
        ssh.run(("/bin/echo", "42"))


def test_ssh_deadline_is_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spawned = Spawned(monkeypatch)
    monkeypatch.setattr(_remote, "DEADLINE_SECONDS", 0.5)
    ssh, _ = fake_ssh(tmp_path, "sleep")
    with pytest.raises(Unavailable, match="deadline"):
        ssh.run(("/bin/echo", "never"))
    spawned.assert_reaped()


def test_ssh_output_bound_applies_after_the_noise(tmp_path: Path) -> None:
    ssh, _ = fake_ssh(tmp_path)
    assert ssh.run(("/bin/sh", "-c", "head -c 10 /dev/zero"), max_stdout=10).stdout == b"\0" * 10
    with pytest.raises(Unavailable, match="byte bound"):
        ssh.run(("/bin/sh", "-c", "head -c 11 /dev/zero"), max_stdout=10)


def test_ssh_client_environment_drops_scheduler_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SBATCH_PARTITION", "hostile")
    monkeypatch.setenv("SLURM_CONF", "/hostile")
    monkeypatch.setenv("HOME", "/home/servatus-test")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent.sock")
    ssh, record = fake_ssh(tmp_path)
    ssh.run(("/bin/true",))
    environment: dict[str, str] = recorded(record)["env"]
    environment.pop("__CF_USER_TEXT_ENCODING", None)  # injected by macOS for every process
    assert set(environment) <= {
        "PATH",
        "LANG",
        "LC_ALL",
        "HOME",
        "LOGNAME",
        "USER",
        "SSH_AUTH_SOCK",
    }
    assert environment["PATH"] == "/usr/bin:/bin"
    assert environment["HOME"] == "/home/servatus-test"
    assert environment["SSH_AUTH_SOCK"] == "/tmp/agent.sock"


def test_scheduler_over_noisy_ssh_reads_exact_replies(tmp_path: Path) -> None:
    # End to end: the fake ssh prints a banner and shell noise, then really executes stub Slurm
    # commands under the scrubbed remote environment.
    slurm = tmp_path / "slurm"
    slurm.mkdir()
    replies = {
        "sbatch": "printf '42\\n'",
        "squeue": "printf 'slurm_load_jobs error: Invalid job id specified\\n' >&2; exit 1",
        "sacct": f"printf '%s' '{sacct_row('COMPLETED', end='2030-01-01T13:00:00')}'",
    }
    for name, body in replies.items():
        stub = slurm / name
        stub.write_text(f"#!/bin/sh\n{body}\n")
        stub.chmod(0o700)
    ssh, _ = fake_ssh(tmp_path)
    scheduler = Scheduler(ssh, str(slurm))
    argv = (str(slurm / "sbatch"), "--parsable", f"--job-name={IDENTITY}")
    assert scheduler.submit(argv, b"#!/bin/sh\n") == JobRef(42)
    observed = scheduler.observe([query()])[ALLOCATION]
    assert observed.allocation.state is AllocationState.SUCCEEDED
