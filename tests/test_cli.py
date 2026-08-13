from __future__ import annotations

import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest

from servatus import Campaign, LogSnapshot, ObservationError, Profile, SlurmTarget, Task, _slurm
from servatus.cli import main


def write_inputs(tmp_path: Path) -> Path:
    stdin = tmp_path / "stdin.bin"
    stdin.write_bytes(b"opaque\x00payload")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"key": "one", "args": ["run"], "stdin_file": stdin.name}) + "\n")
    (tmp_path / "SERVATUS.toml").write_text(
        'default_profile = "cpu"\n'
        "[profiles.cpu.target]\n"
        'host = "login.example.edu"\n'
        'slurm_bin = "/opt/slurm/bin"\n'
        'apptainer = "/usr/bin/apptainer"\n'
        'image = "/images/work.sif"\n'
        'work_root = "/work/project"\n'
        'log_root = "/logs/project"\n'
        'partitions = ["cpu"]\n'
        "max_tasks_per_allocation = 4\n"
        "max_cpus_per_allocation = 16\n"
        "max_memory_mib_per_allocation = 8192\n"
        "max_gpus_per_allocation = 0\n"
        'max_time_limit = "1-00:00:00"\n'
        "max_allocations_per_submit = 4\n"
        "max_script_bytes = 1048576\n"
        "[profiles.cpu.resources]\n"
        "cpus_per_task = 2\n"
        "memory_mib_per_task = 1024\n"
        "gpus_per_task = 0\n"
        'time_limit = "00:10:00"\n'
    )
    return tasks


def plan_arguments(tasks: Path, campaign: Path, output: Path, *extra: str) -> list[str]:
    return [
        "plan",
        str(tasks),
        "--campaign",
        str(campaign),
        "--output",
        str(output),
        *extra,
    ]


def terminal_observations(
    _target: SlurmTarget, queries: tuple[_slurm._AttemptQuery, ...]
) -> tuple[_slurm.SchedulerObservation, ...]:
    return tuple(
        _slurm.SchedulerObservation(
            _slurm.AllocationState.SUCCEEDED,
            "COMPLETED",
            "COMPLETED",
            "0:0",
            None,
            None,
            None,
        )
        for _query in queries
    )


def test_cli_plan_uses_only_cwd_profile_and_inspect_replaces_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tasks = write_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    campaign = tmp_path / "campaign"
    output = tmp_path / "PLAN.json"

    assert main(plan_arguments(tasks, campaign, output)) == 0
    document = json.loads(output.read_text())
    assert document["profile_label"] == "cpu"
    assert document["selected"] == ["one"]
    assert "completed" not in document
    assert "opaque" not in output.read_text()
    capsys.readouterr()

    assert main(["inspect", str(campaign)]) == 0
    view = json.loads(capsys.readouterr().out)
    assert view["tasks"][0]["key"] == "one"
    assert view["tasks"][0]["result"] == "UNOBSERVED"

    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "inspect" in help_text
    assert "status" not in help_text


def test_cli_plan_does_not_search_parent_for_servatus_toml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tasks = write_inputs(tmp_path)
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)

    with pytest.raises(SystemExit) as rejected:
        main(plan_arguments(tasks, child / "campaign", child / "PLAN.json"))

    assert rejected.value.code == 2
    assert "SERVATUS.toml" in capsys.readouterr().err


def test_cli_explicit_profile_overrides_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = write_inputs(tmp_path)
    config = tmp_path / "SERVATUS.toml"
    config.write_text(
        config.read_text()
        + config.read_text()
        .replace('default_profile = "cpu"\n', "")
        .replace("profiles.cpu", "profiles.other")
        .replace('image = "/images/work.sif"', 'image = "/images/other.sif"')
    )
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "PLAN.json"

    main(plan_arguments(tasks, tmp_path / "campaign", output, "--profile", "other"))

    document = json.loads(output.read_text())
    assert document["profile_label"] == "other"
    assert document["target"]["image"] == "/images/other.sif"


def test_cli_plan_output_is_owner_only_and_no_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tasks = write_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "plans" / "PLAN.json"
    arguments = plan_arguments(tasks, tmp_path / "campaign", output)

    previous_umask = os.umask(0)
    try:
        main(arguments)
    finally:
        os.umask(previous_umask)
    original = output.read_bytes()
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    capsys.readouterr()

    with pytest.raises(SystemExit) as duplicate:
        main(arguments)
    assert duplicate.value.code == 2
    assert "destination already exists" in capsys.readouterr().err
    assert output.read_bytes() == original


def test_cli_unknown_retry_prints_duplicate_risk_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tasks_path = write_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    campaign_path = tmp_path / "campaign"
    campaign = Campaign.open(campaign_path, (Task("one", ("run",), b"opaque\x00payload"),))
    selected_profile = Profile.load(tmp_path / "SERVATUS.toml")
    first = campaign.plan(selected_profile, view=campaign.inspect(scheduler=False))
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, b"42;alpha\n", b""),
    )
    campaign.submit(first)
    monkeypatch.setattr(
        _slurm,
        "query_attempts",
        lambda _target, queries: tuple(
            _slurm.SchedulerObservation(
                _slurm.AllocationState.UNKNOWN,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            for _query in queries
        ),
    )
    capsys.readouterr()

    main(
        plan_arguments(
            tasks_path,
            campaign_path,
            tmp_path / "RETRY.json",
            "--retry",
            "one",
            "--allow-duplicate-risk",
            "one",
        )
    )

    assert "duplicate execution risk" in capsys.readouterr().err


def test_cli_validate_submit_seal_and_reconcile_remain_thin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tasks = write_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    campaign_path = tmp_path / "campaign"
    output = tmp_path / "PLAN.json"
    main(plan_arguments(tasks, campaign_path, output))
    capsys.readouterr()

    calls: list[tuple[str, ...]] = []

    def fake(_target: SlurmTarget, argv: tuple[str, ...], _script: bytes) -> _slurm.Result:
        calls.append(argv)
        if argv[-1] == "--test-only":
            return _slurm.Result(0, b"valid now\n", b"")
        return _slurm.Result(1, b"", b"lost receipt")

    monkeypatch.setattr(_slurm, "_run_ssh", fake)
    assert main(["validate", str(campaign_path), str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["time_specific"] is True
    with pytest.raises(SystemExit):
        main(["submit", str(campaign_path), str(output)])
    capsys.readouterr()

    allocation_id = Campaign.load(campaign_path).inspect(scheduler=False).attempts[-1].allocation_id
    monkeypatch.setattr(
        _slurm,
        "query_identity",
        lambda *_args, **_kwargs: _slurm.IdentityMatch(701, "alpha"),
    )
    assert main(["reconcile", str(campaign_path), allocation_id]) == 0
    assert json.loads(capsys.readouterr().out)["job_id"] == 701
    monkeypatch.setattr(_slurm, "query_attempts", terminal_observations)
    assert main(["inspect", str(campaign_path)]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["scheduler_observed"] is True
    assert inspected["attempts"][0]["allocation"]["state"] == "SUCCEEDED"
    assert main(["seal", str(campaign_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"sealed": True}
    assert any(argv[-1] == "--test-only" for argv in calls)


def test_cli_log_delegates_one_exact_raw_byte_snapshot_without_newline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfdbinary: pytest.CaptureFixture[bytes],
) -> None:
    campaign_path = tmp_path / "campaign"
    Campaign.open(campaign_path, ())
    calls: list[tuple[str, str | None, int]] = []

    def read_log(
        self: Campaign,
        allocation_id: str,
        *,
        task_key: str | None = None,
        max_bytes: int = 65_536,
    ) -> LogSnapshot:
        calls.append((allocation_id, task_key, max_bytes))
        return LogSnapshot(b"\x00raw\xff", True, datetime.now(UTC))

    monkeypatch.setattr(Campaign, "read_log", read_log)

    assert main(["log", str(campaign_path), "alloc", "--task", "one", "--bytes", "9"]) == 0

    captured = capfdbinary.readouterr()
    assert captured.out == b"\x00raw\xff"
    assert calls == [("alloc", "one", 9)]


def test_cli_log_help_warns_and_errors_use_existing_redacted_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    campaign_path = tmp_path / "campaign"
    Campaign.open(campaign_path, ())
    with pytest.raises(SystemExit) as help_exit:
        main(["log", "--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "untrusted raw log bytes" in help_text
    assert "redirect" in help_text.lower()

    def unavailable(*_args: object, **_kwargs: object) -> LogSnapshot:
        raise ObservationError("campaign log is unavailable")

    monkeypatch.setattr(Campaign, "read_log", unavailable)
    with pytest.raises(SystemExit) as rejected:
        main(["log", str(campaign_path), "alloc"])
    assert rejected.value.code == 2
    error = capsys.readouterr().err
    assert "campaign log is unavailable" in error
    assert "Traceback" not in error
