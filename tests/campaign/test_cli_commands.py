"""The ``servatus`` CLI, driven in-process against a FakeScheduler."""

from __future__ import annotations

import json
import re
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from servatus import __version__
from servatus.campaign import Campaign, Completed
from servatus.cli import main
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


class Cli:
    """Run commands from inside a synthetic project directory."""

    def __init__(self, root: Path, capsys: pytest.CaptureFixture[str]) -> None:
        self.root = root
        self.capsys = capsys
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

    def __call__(self, *argv: str) -> tuple[int, str, str]:
        code = main(list(argv), connect=self.fake)
        captured = self.capsys.readouterr()
        return code, captured.out, captured.err

    def json(self, *argv: str) -> Any:
        code, out, err = self(*argv, "--json")
        assert code == 0, err
        return json.loads(out)


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> Cli:
    monkeypatch.chdir(tmp_path)
    return Cli(tmp_path, capsys)


def test_version_help_and_usage_errors(cli: Cli) -> None:
    assert cli("--version") == (0, f"servatus {__version__}\n", "")
    code, out, _ = cli("--help")
    assert code == 0 and "mark-not-submitted" in out
    code, _, err = cli("plan")
    assert code == 2 and "usage:" in err
    code, _, err = cli("plan", "state", "--retry", "one", "--retry-failed")
    assert code == 2 and "not allowed with argument" in err


def test_human_workflow_from_creation_to_status(cli: Cli) -> None:
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
    cli.fake.finish_step(1000, 1, "FAILED", exit_code="3:0")
    code, out, _ = cli("status", "state")
    assert code == 0
    assert "one  UNOBSERVED  RUNNING    0:0" in out and "two  UNOBSERVED  FAILED     3:0" in out
    assert "counts: tasks 2, unobserved 2, running 1, failed 1" in out
    assert "next: servatus plan state --retry-failed --output PLAN.json" in out
    cli.fake.finish(1000)
    code, out, _ = cli("status", "state", "--offline")
    assert "ACCEPTED" in out and "scheduler not observed" in out


def test_machine_output_for_every_data_command(cli: Cli) -> None:
    created = cli.json("create", "state", "tasks.jsonl")
    assert created["tasks"] == 2 and created["sealed"] is True
    plan = cli.json("plan", "state", "--output", "plan.json")
    assert plan["format"] == "servatus.plan/1" and plan["selected"] == ["one", "two"]
    assert plan["saved"] == "plan.json" and plan["warnings"] == []
    allocation = plan["allocations"][0]
    assert "script" not in allocation and allocation["cpus"] == 4
    checks = cli.json("validate", "state", "plan.json")["checks"]
    assert [check["accepted"] for check in checks] == [True]
    submitted = cli.json("submit", "state", "plan.json")
    assert submitted["complete"] is True and submitted["unresolved"] == []
    receipt = submitted["receipts"][0]
    assert receipt["job_id"] == 1000 and receipt["task_keys"] == ["one", "two"]
    status = cli.json("status", "state")
    assert status["format"] == "servatus.status/1" and status["counts"]["queued"] == 2
    cancelled = cli.json("cancel", "state", "--task", "one")
    assert [item["job_id"] for item in cancelled["cancelled"]] == [1000]
    doctor = cli.json("doctor")
    assert doctor == {
        "profile": "cpu",
        "host": "login.example.edu",
        "slurm_bin": "/opt/slurm/bin",
        "sbatch_version": "slurm 23.11.4",
        "tasks_per_allocation": 4,
    }


def test_ensure_append_and_task_files(cli: Cli) -> None:
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
        ({"key": "x", "args": [], "env": ["A=1"]}, "env must be an object"),
        ({"key": "x", "args": [], "env": {"1A": "x"}}, "Task.env names must be identifiers"),
        ({"key": "x", "args": [], "env": {"A": 1}}, r"Task.env\[A\] must be a string"),
        ({"key": "x", "args": [], "stdin_file": ""}, "stdin_file must be a nonempty path"),
        ({"key": "x", "args": [], "stdin_file": "missing.bin"}, "cannot read stdin_file"),
        ({"key": "x", "args": "run"}, "args must be an array"),
        ({"key": 1, "args": []}, "Task.key must be a string"),
        ({"key": "x"}, "missing fields: args"),
        ({"key": "x", "args": [], "extra": 1}, "unknown fields: extra"),
        ([], "expected an object"),
    ],
)
def test_invalid_task_files_never_create_a_campaign(cli: Cli, line: object, message: str) -> None:
    cli.tasks({"key": "fine", "args": []}, line, name="bad.jsonl")
    code, out, err = cli("create", "state", "bad.jsonl")
    assert (code, out) == (1, "")
    assert err.startswith("servatus: error: invalid task file bad.jsonl line 2: ")
    assert err.count("\n") == 1 and "usage" not in err
    assert re.search(message, err)
    assert not (cli.root / "state").exists()


def test_duplicate_fields_and_unreadable_files_are_errors(cli: Cli) -> None:
    (cli.root / "dup.jsonl").write_text('{"key": "a", "key": "b", "args": []}\n')
    assert cli("create", "state", "dup.jsonl")[2].endswith("duplicate field 'key'\n")
    assert "cannot read task file absent.jsonl" in cli("create", "state", "absent.jsonl")[2]
    assert cli("status", "nowhere")[:2] == (1, "")


@pytest.mark.parametrize("failure", ["missing_config", "bad_config", "occupied_output"])
def test_planning_failures_never_author_the_campaign(cli: Cli, failure: str) -> None:
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


def test_profiles_are_selected_explicitly(cli: Cli, tmp_path: Path) -> None:
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


def test_plan_reports_holds_deferrals_and_scripts(cli: Cli) -> None:
    cli("create", "state", "tasks.jsonl")
    cli("plan", "state", "--output", "first.json")
    cli("submit", "state", "first.json")
    cli.fake.finish(1000, "FAILED", exit_code="1:0")
    code, out, err = cli(
        "plan", "state", "--retry", "one", "--show-scripts", "--tasks-per-allocation", "1"
    )
    assert code == 0 and "scripts contain Task arguments" in err
    assert "retry: one" in out and "SUBMITTED: two" in out
    assert "#!/bin/sh" in out and "/opt/slurm/bin/sbatch --parsable" in out
    assert "not saved; pass --output PLAN.json to save it" in out
    shown = cli.json("plan", "state", "--retry-failed", "--only", "two", "--show-scripts")
    assert shown["selected"] == ["two"] and shown["held"] == {"one": "NOT_REQUESTED"}
    assert shown["allocations"][0]["script"].startswith("#!/bin/sh")
    code, _, err = cli("plan", "state", "--retry", "ghost")
    assert code == 1 and "unknown Task keys: 'ghost'" in err


def test_interrupted_submission_exits_3_and_recovery_commands_resolve_it(cli: Cli) -> None:
    cli("create", "state", "tasks.jsonl")
    cli("plan", "state", "--output", "plan.json", "--tasks-per-allocation", "1")
    cli.fake.fail_next("sbatch", Completed(0, b"slurm 23.11.4\n", b""))
    cli.fake.fail_next("sbatch", Completed(1, b"", b"sbatch: error: Socket timed out\n"))
    code, out, err = cli("submit", "state", "plan.json", "--json")
    assert code == 3 and err.startswith("servatus: error: submission interrupted")
    result = json.loads(out)
    assert result["complete"] is False and result["receipts"] == []
    lost = result["unresolved"][0]["allocation_id"]
    assert len(result["unattempted"]) == 1
    code, _, err = cli("reconcile", "state", lost)
    assert code == 1 and "0 jobs" in err
    assert (
        cli("mark-not-submitted", "state", lost)[1]
        == f"allocation {lost}: recorded as not submitted\n"
    )

    cli("plan", "state", "--output", "again.json")
    cli.fake.lose_next_reply("sbatch")  # the ping's reply is lost
    code, _, err = cli("submit", "state", "again.json")
    assert code == 75 and "reply from sbatch was lost" in err
    assert Campaign.open(cli.root / "state").status(scheduler=False).revision == 2

    cli.fake.fail_next("sbatch", Completed(0, b"slurm 23.11.4\n", b""))
    cli.fake.fail_next("sbatch", None)
    code, out, _ = cli("submit", "state", "again.json")
    assert code == 3 and "next: servatus reconcile state" in out
    pending = out.split("UNRESOLVED allocation ")[1].split()[0]
    accepted = cli.json("mark-accepted", "state", pending, "4242", "--cluster", "alpha")
    assert accepted["accepted"][0]["cluster"] == "alpha"
    code, out, _ = cli("status", "state", "--offline")
    assert "next: servatus status state" in out


def test_logs_write_raw_bytes_and_refuse_terminals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    monkeypatch.chdir(tmp_path)
    fake = FakeScheduler()
    (tmp_path / "tasks.jsonl").write_text('{"key": "one", "args": ["/bin/true"]}\n')
    (tmp_path / "SERVATUS.toml").write_text(PROFILE)
    assert main(["create", "state", "tasks.jsonl"], connect=fake) == 0
    assert main(["plan", "state", "--output", "plan.json"], connect=fake) == 0
    assert main(["submit", "state", "plan.json"], connect=fake) == 0
    allocation = fake.jobs[0].name.removeprefix("servatus-")
    fake.write_log(f"/logs/{allocation}-1000-0.out", b"\xff\x00raw\x1b[2J")
    capsysbinary.readouterr()
    assert main(["logs", "state", "--task", "one", "--bytes", "5"], connect=fake) == 0
    captured = capsysbinary.readouterr()
    assert captured.out == b"raw\x1b[2J"[-5:]
    assert captured.err == b"servatus: showing the last 5 bytes; earlier output exists\n"
    assert main(["logs", "state", "--task", "one", "--output", "one.log"], connect=fake) == 0
    assert (tmp_path / "one.log").read_bytes() == b"\xff\x00raw\x1b[2J"
    assert stat.S_IMODE((tmp_path / "one.log").stat().st_mode) == 0o600
    assert main(["logs", "state", "--task", "one", "--output", "one.log"], connect=fake) == 1
    tails = fake.count("tail")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    capsysbinary.readouterr()
    assert main(["logs", "state", "--allocation", allocation], connect=fake) == 1
    assert b"refusing to write raw log bytes to a terminal" in capsysbinary.readouterr().err
    assert fake.count("tail") == tails


def test_cancel_needs_a_selector_and_doctor_reports_unavailable_clusters(cli: Cli) -> None:
    cli("create", "state", "tasks.jsonl")
    code, _, err = cli("cancel", "state")
    assert code == 1 and "cancel needs Task keys or allocation ids" in err
    code, out, _ = cli("doctor")
    assert code == 0 and "sbatch: slurm 23.11.4" in out and "up to 4 Tasks" in out
    cli.fake.fail_next("sbatch", Completed(127, b"", b"sbatch: not found\n"))
    code, _, err = cli("doctor")
    assert (code, err) == (75, "servatus: error: sbatch --version failed with exit status 127\n")
