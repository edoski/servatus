from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from test_planning import profile_text

from servatus import Campaign, LogSnapshot, ObservationError, _slurm
from servatus.cli import main


def inputs(path: Path) -> Path:
    (path / "stdin.bin").write_bytes(b"opaque\x00payload")
    tasks = path / "tasks.jsonl"
    tasks.write_text(json.dumps({"key": "one", "args": ["run"], "stdin_file": "stdin.bin"}) + "\n")
    (path / "SERVATUS.toml").write_text(profile_text())
    return tasks


def test_explicit_cli_authoring_plan_validate_submit_and_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    tasks = inputs(tmp_path)
    campaign_path = tmp_path / "campaign"
    plan_path = tmp_path / "plan.json"
    assert main(["create", str(campaign_path), str(tasks), "--appendable"]) == 0
    assert not Campaign.load(campaign_path).inspect(scheduler=False).sealed
    assert main(["seal", str(campaign_path)]) == 0
    before = (campaign_path / "campaign.json").read_bytes()
    assert main(["plan", str(campaign_path), "--output", str(plan_path)]) == 0
    assert (campaign_path / "campaign.json").read_bytes() == before
    assert stat.S_IMODE(plan_path.stat().st_mode) == 0o600
    commands = []

    def run(_target, argv, _script):
        commands.append(argv)
        return _slurm.Result(0, b"valid" if "--test-only" in argv else b"42;alpha\n", b"")

    monkeypatch.setattr(_slurm, "_run_ssh", run)
    assert main(["validate", str(campaign_path), str(plan_path)]) == 0
    assert (campaign_path / "campaign.json").read_bytes() == before
    capsys.readouterr()
    assert main(["submit", str(campaign_path), str(plan_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["receipts"][0]["job_id"] == 42 and result["stop_reason"] is None
    assert commands[0][-1] == "--test-only" and "--test-only" not in commands[1]
    monkeypatch.setattr(
        _slurm,
        "query_attempts",
        lambda _target, queries: tuple(
            _slurm.SchedulerObservation(
                _slurm.AllocationState.SUCCEEDED, "COMPLETED", "COMPLETED", "0:0", None, None, None
            )
            for _ in queries
        ),
    )
    assert main(["status", str(campaign_path)]) == 0
    view = json.loads(capsys.readouterr().out)
    assert view["quiescent"] and not view["results_ready"]


def test_cli_append_registers_only_new_input_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    task_file = inputs(tmp_path)
    campaign_path = tmp_path / "campaign"
    main(["create", str(campaign_path), str(task_file), "--appendable"])
    task_file.write_text(json.dumps({"key": "two", "args": [], "stdin_file": "stdin.bin"}) + "\n")
    main(["append", str(campaign_path), str(task_file)])
    assert [task.key for task in Campaign.load(campaign_path).tasks] == ["one", "two"]
    with pytest.raises(SystemExit):
        main(["append", str(campaign_path), str(task_file)])


@pytest.mark.parametrize("failure", ["missing_config", "occupied_output", "bad_config"])
def test_cli_planning_failure_never_authors_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    monkeypatch.chdir(tmp_path)
    task_file = inputs(tmp_path)
    campaign_path = tmp_path / "campaign"
    main(["create", str(campaign_path), str(task_file)])
    before = (campaign_path / "campaign.json").read_bytes()
    output = tmp_path / "plan.json"
    if failure == "missing_config":
        (tmp_path / "SERVATUS.toml").unlink()
    elif failure == "bad_config":
        (tmp_path / "SERVATUS.toml").write_text("broken TOML")
    else:
        output.write_bytes(b"preserved")
    with pytest.raises(SystemExit):
        main(["plan", str(campaign_path), "--output", str(output)])
    assert (campaign_path / "campaign.json").read_bytes() == before
    if failure == "occupied_output":
        assert output.read_bytes() == b"preserved"


def test_cli_profile_is_explicit_and_never_searches_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_file = inputs(tmp_path)
    (tmp_path / "SERVATUS.toml").write_text(profile_text(second=True))
    campaign_path = tmp_path / "campaign"
    main(["create", str(campaign_path), str(task_file)])
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    with pytest.raises(SystemExit):
        main(["plan", str(campaign_path), "--output", str(child / "plan.json")])
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "plan.json"
    main(["plan", str(campaign_path), "--profile", "alias", "--output", str(output)])
    assert json.loads(output.read_bytes())["profile"]["label"] == "alias"


def test_cli_partial_result_and_operator_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    task_file = inputs(tmp_path)
    campaign_path = tmp_path / "campaign"
    plan_path = tmp_path / "plan.json"
    main(["create", str(campaign_path), str(task_file)])
    main(["plan", str(campaign_path), "--output", str(plan_path)])
    capsys.readouterr()
    monkeypatch.setattr(_slurm, "_run_ssh", lambda *_: _slurm.Result(1, b"", b"lost"))
    assert main(["submit", str(campaign_path), str(plan_path)]) == 1
    result = json.loads(capsys.readouterr().out)
    allocation_id = result["unresolved"][0]["allocation_id"]
    monkeypatch.setattr(
        _slurm, "query_identity", lambda *_args, **_kwargs: _slurm.IdentityMatch(42, "alpha")
    )
    assert main(["reconcile", str(campaign_path), allocation_id]) == 0
    assert json.loads(capsys.readouterr().out)["job_id"] == 42
    assert (
        main(["resolve", str(campaign_path), allocation_id, "--job-id", "42", "--cluster", "alpha"])
        == 0
    )


def test_cli_logs_are_exact_binary_and_errors_are_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    from datetime import UTC, datetime

    campaign_path = tmp_path / "campaign"
    Campaign.create(campaign_path, ())
    calls = []

    def read(self, allocation_id, *, task_key, max_bytes):
        calls.append((allocation_id, task_key, max_bytes))
        return LogSnapshot(b"\xff\0sensitive\x1b", False, datetime.now(UTC))

    monkeypatch.setattr(Campaign, "read_log", read)
    assert main(["logs", str(campaign_path), "abc", "--task", "one", "--bytes", "17"]) == 0
    assert capsysbinary.readouterr().out == b"\xff\0sensitive\x1b"
    assert calls == [("abc", "one", 17)]

    def fail(*_args, **_kwargs):
        raise ObservationError("campaign log is unavailable")

    monkeypatch.setattr(Campaign, "read_log", fail)
    with pytest.raises(SystemExit):
        main(["logs", str(campaign_path), "abc"])
    output = capsysbinary.readouterr()
    assert output.out == b"" and b"campaign log is unavailable" in output.err
