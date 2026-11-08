"""The ``servatus`` CLI, driven in-process against a FakeScheduler."""

from __future__ import annotations

import json
import re
import stat
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Generic, TypeVar

import pytest

from servatus import __version__
from servatus.campaign import Campaign, Completed, PlannedAllocation, Profile
from servatus.cli import main
from servatus.errors import SubmissionInterrupted
from servatus.testing import FakeScheduler

PROFILE = """\
default_profile = "cpu"

[target]
host = "login.example.edu"
slurm_bin = "/opt/slurm/bin"
work_root = "/work"
log_root = "/logs"
partitions = ["cpu"]
max_tasks_per_allocation = 4
max_cpus_per_allocation = 16
max_memory_mib_per_allocation = 8192
max_time_limit = "1-00:00:00"

[resources]
cpus = 2
memory_mib = 1024
time_limit = "00:10:00"

[profiles.cpu]

[profiles.wide.target]
partitions = ["wide"]
"""


Output = TypeVar("Output", str, bytes)


class Cli(Generic[Output]):
    """Run commands from inside a synthetic project directory, capturing text or bytes."""

    def __init__(self, root: Path, capture: pytest.CaptureFixture[Output]) -> None:
        self.root = root
        self.capture = capture
        self.fake = FakeScheduler()
        (root / "stdin.bin").write_bytes(b"opaque\x00payload")
        (root / "SERVATUS.toml").write_text(PROFILE)
        self.tasks(
            {"key": "one", "args": ["/usr/bin/python3", "train.py"], "stdin_file": "stdin.bin"},
            {"key": "two", "args": ["/usr/bin/python3", "train.py"], "env": {"MODE": "smoke"}},
        )

    def tasks(self, *lines: object, name: str = "tasks.jsonl") -> Path:
        path = self.root / name
        path.write_text("".join(json.dumps(line) + "\n" for line in lines))
        return path

    def __call__(self, *argv: str) -> tuple[int, Output, Output]:
        code = main(list(argv), connect=self.fake)
        captured = self.capture.readouterr()
        return code, captured.out, captured.err

    def submit_failing_twenty_minutes_ago(self, **plan: Any) -> PlannedAllocation:
        """An API caller plans and submits; its first ``sbatch`` fails and its intent is old
        enough for ``reconcile`` to decide now."""
        earlier = Campaign.open(
            self.root / "state",
            connect=self.fake,
            clock=lambda: datetime.now(UTC) - timedelta(minutes=20),
        )
        planned = earlier.plan(Profile.load(self.root / "SERVATUS.toml"), **plan)
        self.fake.fail_next("sbatch")
        with pytest.raises(SubmissionInterrupted, match="unresolved"):
            earlier.submit(planned)
        return planned.allocations[0]

    def json(self, *argv: str) -> Any:
        code, out, err = self(*argv, "--json")
        assert code == 0, err
        return json.loads(out)


@pytest.fixture
def cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> Cli[str]:
    monkeypatch.chdir(tmp_path)
    return Cli(tmp_path, capsys)


@pytest.fixture
def raw_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
) -> Cli[bytes]:
    monkeypatch.chdir(tmp_path)
    return Cli(tmp_path, capsysbinary)


def test_version_help_and_usage_errors(cli: Cli[str]) -> None:
    assert cli("--version") == (0, f"servatus {__version__}\n", "")
    code, out, _ = cli("--help")
    assert code == 0 and "reconcile" in out and "mark-" not in out
    code, _, err = cli("plan")
    assert code == 2 and "usage:" in err
    code, _, err = cli("plan", "state", "--retry", "one", "--retry-failed")
    assert code == 2 and "not allowed with argument" in err


@pytest.mark.parametrize(
    "argv",
    [
        ("mark-accepted", "state", "a" * 24, "1000"),
        ("mark-not-submitted", "state", "a" * 24),
        ("plan", "state", "--retry", "one", "--allow-duplicate-risk", "one"),
        ("reconcile", "state", "a" * 24),
    ],
)
def test_removed_commands_and_options_are_usage_errors(
    cli: Cli[str], argv: tuple[str, ...]
) -> None:
    cli("create", "state", "tasks.jsonl")
    code, out, err = cli(*argv)
    assert (code, out) == (2, "") and "usage:" in err


def test_human_workflow_from_creation_to_status(cli: Cli[str]) -> None:
    code, out, err = cli("create", "state", "tasks.jsonl", "--appendable")
    assert (code, err) == (0, "") and out.startswith("created campaign ")
    assert "2 Tasks, appendable" in out
    assert cli("seal", "state")[1].startswith("sealed campaign")
    before = (cli.root / "state" / "campaign.json").read_bytes()

    code, out, _ = cli("plan", "state", "--output", "plan.json")
    assert code == 0
    assert "selected: 2 Tasks in 1 allocation" in out and "saved: plan.json" in out
    assert (cli.root / "state" / "campaign.json").read_bytes() == before
    assert stat.S_IMODE((cli.root / "plan.json").stat().st_mode) == 0o600

    code, out, _ = cli("validate", "state", "plan.json")
    assert code == 0 and "2 Tasks (4 CPUs, 2048 MiB, 0 GPUs, 00:10:00): accepted" in out
    assert (cli.root / "state" / "campaign.json").read_bytes() == before

    code, out, _ = cli("submit", "state", "plan.json")
    assert code == 0 and "submitted allocation" in out and "job 1000 (one, two)" in out
    assert cli.fake.jobs[0].script.count(b"opaque\\000payload") == 1

    cli.fake.start(1000)
    cli.fake.finish_step(1000, 1, exit_code=3)
    code, out, _ = cli("status", "state")
    assert code == 0
    assert re.search(r"^KEY  RESULT      EXECUTION  EXIT  RUN TIME   LINK  ALLOCATION$", out, re.M)
    id_ = r"[0-9a-f]{24}$"
    assert re.search(rf"^one  UNOBSERVED  RUNNING    -     ~00:00:00  -     {id_}", out, re.M)
    assert re.search(rf"^two  UNOBSERVED  FAILED     3     00:00:00   -     {id_}", out, re.M)
    assert "counts: tasks 2, unobserved 2, running 1, failed 1" in out
    assert "next: servatus plan state --retry-failed --output PLAN.json" in out
    cli.fake.finish(1000)
    code, out, _ = cli("status", "state", "--offline")
    assert "ACCEPTED" in out and "scheduler not observed" in out


def test_machine_output_for_every_data_command(cli: Cli[str]) -> None:
    created = cli.json("create", "state", "tasks.jsonl")
    assert created["tasks"] == 2 and created["sealed"] is True
    plan = cli.json("plan", "state", "--output", "plan.json")
    assert plan["format"] == "servatus.plan/2" and plan["selected"] == ["one", "two"]
    assert plan["saved"] == "plan.json" and plan["warnings"] == []
    allocation = plan["allocations"][0]
    assert "script" not in allocation and allocation["cpus"] == 4
    checks = cli.json("validate", "state", "plan.json")["checks"]
    assert [check["accepted"] for check in checks] == [True]
    submitted = cli.json("submit", "state", "plan.json")
    assert submitted["complete"] is True and submitted["unresolved"] == []
    receipt = submitted["receipts"][0]
    assert receipt["job"] == 1000
    assert receipt["task_keys"] == ["one", "two"]
    status = cli.json("status", "state")
    assert status["format"] == "servatus.status/3" and status["counts"]["queued"] == 2
    assert status["warnings"] == [] and status["tasks"][0]["exit_code"] is None
    cancelled = cli.json("cancel", "state", "--task", "one")
    assert cancelled == {
        "cancelled": [
            {"allocation_id": receipt["allocation_id"], "job": 1000, "task_keys": ["one", "two"]}
        ]
    }
    assert cli.json("cancel", "state", "--task", "two") == {"cancelled": []}
    assert cli.json("reconcile", "state") == {
        "receipts": [],
        "not_submitted": [],
        "waiting": [],
        "complete": True,
    }
    doctor = cli.json("doctor")
    checks = doctor.pop("checks")
    assert doctor == {
        "profile": "cpu",
        "host": "login.example.edu",
        "slurm_bin": "/opt/slurm/bin",
        "tasks_per_allocation": 4,
        "verdict": "pass",
    }
    assert [check["verdict"] for check in checks] == ["pass"] * 8
    assert checks[0] == {"name": "slurm version", "verdict": "pass", "detail": "slurm 23.11.4"}


def test_ensure_append_and_task_files(cli: Cli[str]) -> None:
    assert cli.json("ensure", "state", "tasks.jsonl")["sealed"] is False
    more = cli.tasks({"key": "three", "args": ["/bin/true"]}, name="more.jsonl")
    assert cli.json("append", "state", str(more))["tasks"] == 3
    code, _, err = cli("append", "state", str(more))
    assert code == 1 and err == "servatus: error: appended Task keys already exist: three\n"
    assert cli.json("ensure", "state", "tasks.jsonl")["tasks"] == 3
    first, second, _ = Campaign.open(cli.root / "state").tasks()
    assert first.stdin == b"opaque\x00payload" and dict(second.env) == {"MODE": "smoke"}
    assert cli.json("ensure", "sealed", "tasks.jsonl", "--sealed")["sealed"] is True


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ({"key": "x", "args": [], "env": ["A=1"]}, "Task.env must map environment names"),
        ({"key": "x", "args": [], "env": {"1A": "x"}}, "Task.env names must be identifiers"),
        ({"key": "x", "args": [], "env": {"A": 1}}, r"Task.env\[A\] must be a string"),
        ({"key": "x", "args": [], "stdin_file": ""}, "stdin_file must be a nonempty path"),
        ({"key": "x", "args": [], "stdin_file": "missing.bin"}, "cannot read stdin_file"),
        ({"key": "x", "args": "run"}, "Task.args must be a sequence"),
        ({"key": 1, "args": []}, "Task.key must be a string"),
        ({"key": "x"}, "missing fields: args"),
        ({"key": "x", "args": [], "extra": 1}, "unknown fields: extra"),
        ([], "expected an object"),
    ],
)
def test_invalid_task_files_never_create_a_campaign(
    cli: Cli[str], line: object, message: str
) -> None:
    cli.tasks({"key": "fine", "args": []}, line, name="bad.jsonl")
    code, out, err = cli("create", "state", "bad.jsonl")
    assert (code, out) == (1, "")
    assert err.startswith("servatus: error: invalid task file bad.jsonl line 2: ")
    assert err.count("\n") == 1 and "usage" not in err
    assert re.search(message, err)
    assert not (cli.root / "state").exists()


def test_duplicate_fields_and_unreadable_files_are_errors(cli: Cli[str]) -> None:
    (cli.root / "dup.jsonl").write_text('{"key": "a", "key": "b", "args": []}\n')
    assert cli("create", "state", "dup.jsonl")[2].endswith("duplicate field 'key'\n")
    assert "cannot read task file absent.jsonl" in cli("create", "state", "absent.jsonl")[2]
    assert cli("status", "nowhere")[:2] == (1, "")


def test_task_lines_end_only_at_newlines(cli: Cli[str]) -> None:
    # JSON allows raw U+2028, U+0085, and friends inside strings; they are not line breaks.
    (cli.root / "odd.jsonl").write_text(
        '{"key": "a b\x85c", "args": []}\r\n\n{"key": "d", "args": []}\n', newline=""
    )
    assert cli.json("create", "state", "odd.jsonl")["tasks"] == 2
    assert [task.key for task in Campaign.open(cli.root / "state").tasks()] == [
        "a b\x85c",
        "d",
    ]


def test_runaway_nesting_is_an_input_error(cli: Cli[str]) -> None:
    (cli.root / "deep.jsonl").write_text("[" * 100_000 + "]" * 100_000 + "\n")
    code, _, err = cli("create", "state", "deep.jsonl")
    assert code == 1 and err.startswith("servatus: error: invalid task file deep.jsonl line 1")
    cli("create", "state", "tasks.jsonl")
    (cli.root / "deep.json").write_text('{"held": ' + "[" * 100_000 + "]" * 100_000 + "}")
    code, _, err = cli("submit", "state", "deep.json")
    assert code == 1 and err.startswith("servatus: error: invalid plan document")
    assert err.count("\n") == 1


def test_unexpected_failures_are_one_line_and_interrupts_exit_130(
    cli: Cli[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    cli("create", "state", "tasks.jsonl")

    def broken(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom\x1b[2J")

    monkeypatch.setattr(Campaign, "status", broken)
    assert cli("status", "state") == (
        1,
        "",
        "servatus: error: unexpected RuntimeError: boom\\x1b[2J\n",
    )

    def interrupted(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(Campaign, "status", interrupted)
    assert cli("status", "state") == (130, "", "servatus: error: interrupted\n")
    code, _, err = cli("plan", "state", "--config", "bad\0name.toml")
    assert code == 1 and err.startswith("servatus: error: cannot read TOML configuration")


def test_human_output_escapes_untrusted_text(cli: Cli[str]) -> None:
    cli.tasks({"key": "evil\x1b[2J\r", "args": ["/bin/true"]}, name="evil.jsonl")
    cli("create", "state", "evil.jsonl")
    code, out, _ = cli("plan", "state", "--output", "plan.json")
    assert code == 0 and "evil\\x1b[2J\\r" in out and "\x1b" not in out
    cli.fake.fail_next("sbatch", Completed(1, b"", b"sbatch: error: \x1b]0;pwned\x07\n"))
    code, out, _ = cli("validate", "state", "plan.json")
    assert code == 1 and "sbatch: error: \\x1b]0;pwned\\x07" in out and "\x1b" not in out
    code, out, err = cli("submit", "state", "plan.json")
    assert code == 0 and "(evil\\x1b[2J\\r)" in out and "\x1b" not in out + err
    code, out, _ = cli("status", "state")
    assert code == 0 and "evil\\x1b[2J\\r  " in out and "\x1b" not in out
    assert cli.json("status", "state")["tasks"][0]["key"] == "evil\x1b[2J\r"
    cli.fake.fail_next("ping", Completed(0, b"slurm \x1b[31m23\n", b""))
    code, out, _ = cli("doctor")
    assert code == 1 and "slurm \\x1b[31m23" in out and "\x1b" not in out


@pytest.mark.parametrize("failure", ["missing_config", "bad_config", "occupied_output"])
def test_planning_failures_never_author_the_campaign(cli: Cli[str], failure: str) -> None:
    cli("create", "state", "tasks.jsonl")
    before = (cli.root / "state" / "campaign.json").read_bytes()
    output = cli.root / "plan.json"
    if failure == "missing_config":
        (cli.root / "SERVATUS.toml").unlink()
    elif failure == "bad_config":
        (cli.root / "SERVATUS.toml").write_text("broken TOML")
    else:
        output.write_bytes(b"preserved")
    code, out, err = cli("plan", "state", "--output", "plan.json")
    assert (code, out) == (1, "") and err.startswith("servatus: error: ")
    assert (cli.root / "state" / "campaign.json").read_bytes() == before
    if failure == "occupied_output":
        assert output.read_bytes() == b"preserved"


def test_profiles_are_selected_explicitly(cli: Cli[str], tmp_path: Path) -> None:
    cli("create", "state", "tasks.jsonl")
    plan = cli.json("plan", "state", "--profile", "wide")
    assert plan["profile"]["label"] == "wide"
    assert plan["profile"]["target"]["partitions"] == ["wide"]
    elsewhere = tmp_path / "child"
    elsewhere.mkdir()
    code, _, err = cli(
        "plan", str(cli.root / "state"), "--config", str(elsewhere / "SERVATUS.toml")
    )
    assert code == 1 and "configuration file does not exist" in err
    code, _, err = cli("doctor", "--profile", "absent")
    assert code == 1 and "'absent' is not declared" in err


def test_plan_reports_holds_deferrals_and_scripts(cli: Cli[str]) -> None:
    cli("create", "state", "tasks.jsonl")
    cli("plan", "state", "--output", "first.json")
    cli("submit", "state", "first.json")
    cli.fake.finish(1000, exit_code=1)
    cli.fake.purge(1000)
    code, out, err = cli(
        "plan", "state", "--retry", "one", "--show-scripts", "--tasks-per-allocation", "1"
    )
    assert code == 0 and "scripts contain Task arguments" in err
    assert "retry: one" in out and "SUBMITTED: two" in out
    assert "#!/bin/sh" in out and "/opt/slurm/bin/sbatch --parsable" in out
    assert "not saved; pass --output PLAN.json to save it" in out
    shown = cli.json("plan", "state", "--retry-failed", "--only", "two", "--show-scripts")
    assert shown["selected"] == ["two"] and shown["held"] == {"one": "NOT_REQUESTED"}
    assert shown["retry"] == ["two"]
    assert "retry" not in json.loads((cli.root / "first.json").read_bytes())
    assert shown["allocations"][0]["script"].startswith("#!/bin/sh")
    code, _, err = cli("plan", "state", "--retry", "ghost")
    assert code == 1 and "unknown Task keys: 'ghost'" in err


def test_reconcile_reports_each_outcome_and_exits_75_while_waiting(cli: Cli[str]) -> None:
    cli("create", "state", "tasks.jsonl")
    old = cli.submit_failing_twenty_minutes_ago(tasks_per_allocation=1).allocation_id

    cli("plan", "state", "--output", "two.json")
    cli.fake.lose_next_reply("sbatch")
    code, out, err = cli("submit", "state", "two.json", "--json")
    assert code == 3 and err.startswith("servatus: error: submission interrupted")
    submitted = json.loads(out)
    assert submitted["complete"] is False and submitted["receipts"] == []
    found = submitted["unresolved"][0]["allocation_id"]

    code, out, err = cli("reconcile", "state")
    assert (code, err) == (0, "")
    assert out == (
        f"accepted allocation {found}: job 1000 (two)\nnot submitted: allocation {old} (one)\n"
    )
    assert cli("reconcile", "state")[:2] == (0, "nothing to reconcile\n")

    cli("plan", "state", "--output", "again.json")
    cli.fake.fail_next("sbatch")
    code, out, _ = cli("submit", "state", "again.json")
    assert code == 3 and "next: servatus reconcile state" in out
    young = out.split("UNRESOLVED allocation ")[1].split()[0]
    code, out, _ = cli("reconcile", "state")
    assert code == 75
    lines = out.splitlines()
    assert lines[0] == f"waiting: allocation {young} (one)"
    assert re.fullmatch(r"  no evidence yet; reconcile again after \S+\+00:00", lines[1])
    code, out, err = cli("reconcile", "state", "--json")
    assert (code, err) == (75, "")
    document = json.loads(out)
    assert document["complete"] is False and document["receipts"] == []
    (waiting,) = document["waiting"]
    assert (waiting["allocation_id"], waiting["task_keys"]) == (young, ["one"])
    decidable = datetime.fromisoformat(waiting["decidable_at"].replace("Z", "+00:00"))
    assert timedelta(minutes=14) < decidable - datetime.now(UTC) <= timedelta(minutes=15)
    code, out, _ = cli("status", "state")
    assert "next: servatus reconcile state" in out


def test_status_explains_outcomes_warnings_and_next_steps(cli: Cli[str]) -> None:
    cli.tasks(*({"key": key, "args": ["/bin/true"]} for key in "abc"), name="abc.jsonl")
    cli("create", "state", "abc.jsonl", "--appendable")
    cli("plan", "state", "--output", "plan.json", "--tasks-per-allocation", "1")
    cli("submit", "state", "plan.json")
    cli.fake.finish(1000, exit_code=3)
    assert cli.json("cancel", "state", "--task", "b")["cancelled"][0]["job"] == 1001
    cli.fake.start(1002)
    cli.fake.lose(1002)
    for job in (1000, 1001, 1002):
        cli.fake.purge(job)
    code, out, _ = cli("status", "state")
    assert code == 0
    assert re.search(r"^a\s+UNOBSERVED  FAILED\s+3\s+00:00:00\s+-\s+[0-9a-f]{24}$", out, re.M)
    never = r"^b\s+UNOBSERVED  STOPPED \(never started\)  -\s+00:00:00\s+-\s+[0-9a-f]{24}$"
    assert re.search(never, out, re.M)
    assert re.search(r"^c\s+UNOBSERVED  LOST\s+-\s+~00:10:00\s+-\s+[0-9a-f]{24}$", out, re.M)
    assert "counts: tasks 3, unobserved 3, failed 1, stopped 1, lost 1" in out
    assert "quiescent: yes; results ready: no" in out
    assert "next: servatus plan state --retry-failed --output PLAN.json" in out
    assert "next: servatus plan state --retry KEY --output PLAN.json  (LOST Tasks)" in out
    assert "warning:" not in out and "next: servatus reconcile" not in out

    cli.tasks({"key": "d", "args": ["/bin/true"]}, name="d.jsonl")
    cli("append", "state", "d.jsonl")
    stray = cli.submit_failing_twenty_minutes_ago(only=["d"])
    assert cli("reconcile", "state")[0] == 0
    cli.fake.run(stray.argv, stdin=stray.script)
    code, out, _ = cli("status", "state")
    assert code == 0 and "quiescent: no" in out
    warnings = [line for line in out.splitlines() if line.startswith("warning: ")]
    assert warnings == [
        f"warning: allocation {stray.allocation_id} is recorded as not submitted, but Slurm "
        "lists job 1003 for it; its Tasks stay held ACTIVE until it leaves the queue"
    ]
    assert "next: servatus plan state --output PLAN.json" in out
    code, out, _ = cli("plan", "state", "--only", "d")
    assert code == 0 and "ACTIVE: d" in out
    assert cli.json("cancel", "state", "--allocation", stray.allocation_id)["cancelled"] == [
        {"allocation_id": stray.allocation_id, "job": 1003, "task_keys": ["d"]}
    ]


def test_logs_write_raw_bytes_and_refuse_terminals(
    raw_cli: Cli[bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    cli, fake = raw_cli, raw_cli.fake
    for argv in (("create", "state", "tasks.jsonl"), ("plan", "state", "--output", "plan.json")):
        assert cli(*argv)[0] == 0
    assert cli("submit", "state", "plan.json")[0] == 0
    allocation = fake.jobs[0].name.removeprefix("servatus-")
    fake.write_file(f"/logs/{allocation}-1000-0.out", b"\xff\x00raw\x1b[2J")
    assert cli("logs", "state", "--task", "one", "--bytes", "5") == (
        0,
        b"raw\x1b[2J"[-5:],
        b"servatus: showing the last 5 bytes; earlier output exists\n",
    )
    assert cli("logs", "state", "--task", "one", "--output", "one.log")[0] == 0
    assert (cli.root / "one.log").read_bytes() == b"\xff\x00raw\x1b[2J"
    assert stat.S_IMODE((cli.root / "one.log").stat().st_mode) == 0o600
    assert cli("logs", "state", "--task", "one", "--output", "one.log")[0] == 1
    tails = fake.count("tail")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    code, _, err = cli("logs", "state", "--allocation", allocation)
    assert code == 1 and b"refusing to write raw log bytes to a terminal" in err
    assert fake.count("tail") == tails


def test_cancel_needs_a_selector_and_doctor_reports_unavailable_clusters(cli: Cli[str]) -> None:
    cli("create", "state", "tasks.jsonl")
    code, _, err = cli("cancel", "state")
    assert code == 1 and "cancel needs Task keys or allocation ids" in err
    code, out, _ = cli("doctor")
    assert (
        code == 0 and "pass        slurm version: slurm 23.11.4" in out and "up to 4 Tasks" in out
    )
    cli.fake.fail_next("ping", Completed(127, b"", b"sbatch: not found\n"))
    code, out, _ = cli("doctor")
    assert code == 1
    assert "fail        slurm version: sbatch --version failed with exit status 127: sbatch" in out
    cli.fake.fail_next("ping")
    code, out, _ = cli("doctor")
    assert code == 75 and "unavailable slurm version: FakeScheduler" in out
    assert "squeue" not in out


def test_doctor_exits_by_its_worst_verdict(cli: Cli[str]) -> None:
    cli.fake.fail_next("squeue")
    code, out, _ = cli("doctor")
    assert code == 75 and "unavailable squeue: FakeScheduler: injected failure" in out
    cli.fake.fail_next("squeue")
    cli.fake.fail_next("sh", Completed(1, b"", b""))
    code, out, _ = cli("doctor")
    assert code == 1 and "fail        work_root: " in out and "unavailable squeue" in out
    cli.fake.fail_next("sh", Completed(1, b"", b""))
    code, out, _ = cli("doctor", "--json")
    assert code == 1 and json.loads(out)["verdict"] == "fail"
