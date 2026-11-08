"""Leases and steps through the ``servatus`` CLI, driven in-process against a FakeScheduler."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

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

[profiles.lease.resources]
signal_before_end = "00:02:00"
"""
SHAPE = "4 CPUs, 2048 MiB, 0 GPUs, 00:10:00"


class Cli:
    """Run commands from inside a synthetic project directory with an appendable campaign."""

    def __init__(self, root: Path, capture: pytest.CaptureFixture[str]) -> None:
        self.root = root
        self.capture = capture
        self.fake = FakeScheduler()
        (root / "SERVATUS.toml").write_text(PROFILE)
        self.tasks(task("one"), task("two"))
        assert self("create", "state", "tasks.jsonl", "--appendable")[0] == 0

    def tasks(self, *lines: object, name: str = "tasks.jsonl") -> str:
        (self.root / name).write_text("".join(json.dumps(line) + "\n" for line in lines))
        return name

    def __call__(self, *argv: str) -> tuple[int, str, str]:
        code = main(list(argv), connect=self.fake)
        captured = self.capture.readouterr()
        return code, captured.out, captured.err

    def json(self, *argv: str) -> Any:
        code, out, err = self(*argv, "--json")
        assert code == 0, err
        return json.loads(out)

    def lease(self) -> str:
        """Hold and submit a lease of 2 slots that is running as job 1000; its id."""
        plan = self.json("plan", "state", "--hold", "2", "--profile", "lease", "--output", "l.json")
        assert self("submit", "state", "l.json")[0] == 0
        self.fake.start(1000)
        return plan["allocations"][0]["allocation_id"]

    def steps(self) -> dict[str, str]:
        status = self.json("status", "state")
        return {item["key"]: item["current_allocation_id"] for item in status["tasks"]}


def task(key: str, *args: str) -> dict[str, object]:
    return {"key": key, "args": ["/usr/bin/python3", "train.py", *args]}


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> Cli:
    monkeypatch.chdir(tmp_path)
    return Cli(tmp_path, capsys)


def test_a_hold_plan_prints_its_lease_and_submits_one_job(cli: Cli) -> None:
    code, out, err = cli("plan", "state", "--hold", "2", "--profile", "lease", "--output", "l.json")
    assert (code, err) == (0, "")
    lease = re.search(r"^  ([0-9a-f]{24})  (.*)$", out, re.M)
    assert lease is not None and lease.group(2) == SHAPE
    assert "lease: 2 slots of 2 CPUs, 1024 MiB, 0 GPUs, 00:10:00\n" in out
    assert "held: 2 Tasks\n  NOT_REQUESTED: one, two\n" in out and "selected:" not in out
    code, out, _ = cli("validate", "state", "l.json")
    assert code == 0 and out.startswith(f"2 slots ({SHAPE}): accepted\n")
    assert cli("submit", "state", "l.json") == (
        0,
        f"submitted lease {lease.group(1)}: job 1000\n",
        "",
    )
    code, out, _ = cli("plan", "state", "--hold", "1", "--profile", "lease", "--max-links", "2")
    assert code == 0 and out.count(", link ") == 2 and ", link 2/2\n" in out


def test_start_registers_unseen_tasks_and_starts_the_files_keys(cli: Cli) -> None:
    lease = cli.lease()
    code, out, err = cli("start", "state", "tasks.jsonl", "--lease", lease)
    assert (code, err) == (0, "")
    steps = cli.steps()
    assert out == (
        f"submitted allocation {steps['one']}: job 1000 (one)\n"
        f"submitted allocation {steps['two']}: job 1000 (two)\n"
    )
    more = cli.tasks(task("one"), task("three"), name="more.jsonl")
    code, out, _ = cli("start", "state", more, "--lease", lease)
    lines = out.splitlines()
    assert code == 0 and lines[0] == "held one: SUBMITTED"
    assert lines[1:] == [f"submitted allocation {cli.steps()['three']}: job 1000 (three)"]
    assert len(cli.fake.poll(1000)) == 3


@pytest.mark.parametrize(
    ("state", "lines", "message"),
    [
        ("state", (task("one", "changed"),), "Task 'one' differs from the registered Task"),
        ("absent", (task("one"),), "campaign does not exist"),
    ],
)
def test_start_refuses_a_changed_task_or_a_missing_campaign(
    cli: Cli, state: str, lines: tuple[object, ...], message: str
) -> None:
    lease = cli.lease()
    code, out, err = cli("start", state, cli.tasks(*lines, name="bad.jsonl"), "--lease", lease)
    assert (code, out) == (1, "") and err.startswith("servatus: error: ") and message in err
    assert not (cli.root / "absent").exists()
    assert len(cli.fake.jobs) == 1 and cli.fake.count("sh") == 0


def test_start_retry_restarts_a_finished_key_in_the_lease(cli: Cli) -> None:
    lease = cli.lease()
    cli("start", "state", "tasks.jsonl", "--lease", lease)
    first = cli.steps()
    cli.fake.poll(1000)
    cli.fake.end_step(first["one"], 1)
    code, out, _ = cli("start", "state", "tasks.jsonl", "--lease", lease, "--retry", "one")
    again = cli.steps()["one"]
    assert code == 0 and again != first["one"]
    assert out == f"held two: SUBMITTED\nsubmitted allocation {again}: job 1000 (one)\n"


def test_status_shows_steps_in_their_lease_and_one_line_per_lease(cli: Cli) -> None:
    lease = cli.lease()
    cli("start", "state", cli.tasks(task("one"), name="one.jsonl"), "--lease", lease)
    step = cli.steps()["one"]
    cli.fake.poll(1000)
    cli.fake.begin_step(step)
    code, out, _ = cli("status", "state")
    assert code == 0
    assert re.search(rf"^one  UNOBSERVED  RUNNING .* {step} \(lease {lease}\)$", out, re.M)
    assert f"\nlease {lease}: job 1000 RUNNING, 2 slots\n" in out
    document = cli.json("status", "state")
    assert document["format"] == "servatus.status/4"
    attempts = {item["allocation_id"]: item for item in document["attempts"]}
    assert (attempts[lease]["slots"], attempts[lease]["within"]) == (2, None)
    assert (attempts[step]["slots"], attempts[step]["within"]) == (1, lease)


def test_plan_documents_carry_slots_and_the_lease(cli: Cli) -> None:
    held = cli.json("plan", "state", "--hold", "2", "--profile", "lease")
    assert held["format"] == "servatus.plan/4"
    assert (held["slots"], held["within"], held["selected"]) == (2, None, [])
    assert held["allocations"][0]["slots"] == 2 and held["allocations"][0]["task_keys"] == []
    lease = cli.lease()
    steps = cli.json("plan", "state", "--lease", lease, "--slots", "2", "--output", "s.json")
    assert (steps["slots"], steps["within"], steps["selected"]) == (2, lease, ["one", "two"])
    assert [item["cpus"] for item in steps["allocations"]] == [4, 4]
    code, out, _ = cli("plan", "state", "--lease", lease, "--only", "one")
    assert code == 0 and f"in lease {lease}: 1 slot per step\n" in out
    assert "selected: 1 Task in 1 step\n" in out
    assert cli("validate", "state", "s.json") == (
        0,
        "nothing to validate: steps start inside their lease, not through sbatch\n",
        "",
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (("--hold", "2", "--lease", "x"), "--hold takes no --lease"),
        (("--hold", "2", "--only", "one"), "--hold takes no --only"),
        (("--hold", "2", "--retry", "one"), "--hold takes no --retry"),
        (("--hold", "2", "--budget", "01:00:00"), "--hold takes no --budget"),
        (("--hold", "2", "--tasks-per-allocation", "1"), "--hold takes no --tasks-per-allocation"),
        (("--lease", "x", "--profile", "lease"), "--lease takes no --profile"),
        (("--lease", "x", "--config", "SERVATUS.toml"), "--lease takes no --config"),
        (
            ("--lease", "x", "--tasks-per-allocation", "1"),
            "--lease takes no --tasks-per-allocation",
        ),
        (("--slots", "1"), "a plan without --lease takes no --slots"),
    ],
)
def test_plan_refuses_options_that_do_not_belong_to_its_mode(
    cli: Cli, options: tuple[str, ...], message: str
) -> None:
    assert cli("plan", "state", *options) == (1, "", f"servatus: error: {message}\n")
    assert cli.fake.calls == ()


def test_cancel_of_a_task_withdraws_its_step(cli: Cli) -> None:
    lease = cli.lease()
    cli("start", "state", "tasks.jsonl", "--lease", lease)
    step = cli.steps()["one"]
    assert cli("cancel", "state", "--task", "one") == (
        0,
        f"cancelled allocation {step}: job 1000 (one)\n",
        "",
    )
    assert cli.fake.poll(1000) == (cli.steps()["two"],)
