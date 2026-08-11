from __future__ import annotations

import base64
import json
import os
import stat
from pathlib import Path

import pytest

from servatus import SlurmTarget, _slurm
from servatus.cli import main


def write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    stdin = tmp_path / "stdin.bin"
    stdin.write_bytes(b"opaque\x00payload")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"key": "one", "args": ["run"], "stdin_file": stdin.name}) + "\n")
    resources = tmp_path / "resources.toml"
    resources.write_text(
        "cpus_per_task = 2\nmemory_mib_per_task = 1024\ngpus_per_task = 0\n"
        'time_limit = "00:10:00"\n'
    )
    target = tmp_path / "target.toml"
    target.write_text(
        'host = "login.example.edu"\nslurm_bin = "/opt/slurm/bin"\n'
        'apptainer = "/usr/bin/apptainer"\nimage = "/images/work.sif"\n'
        'work_root = "/work/project"\nlog_root = "/logs/project"\npartitions = ["cpu"]\n'
        "max_tasks_per_allocation = 4\nmax_cpus_per_allocation = 16\n"
        "max_memory_mib_per_allocation = 8192\nmax_gpus_per_allocation = 0\n"
        'max_time_limit = "1-00:00:00"\nmax_allocations_per_submit = 4\n'
        "max_script_bytes = 1048576\n"
    )
    return tasks, resources, target


def plan_arguments(
    tasks: Path,
    resources: Path,
    target: Path,
    campaign: Path,
    output: Path,
    *extra: str,
) -> list[str]:
    return [
        "plan",
        str(tasks),
        "--target",
        str(target),
        "--resources",
        str(resources),
        "--campaign",
        str(campaign),
        "--output",
        str(output),
        *extra,
    ]


def test_cli_plan_status_and_help(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tasks, resources, target = write_inputs(tmp_path)
    campaign = tmp_path / "campaign"
    output = tmp_path / "PLAN.json"
    assert main(plan_arguments(tasks, resources, target, campaign, output)) == 0
    document = output.read_text()
    assert "opaque" not in document
    assert "payload" not in document
    assert '"one"' in document
    capsys.readouterr()
    assert main(["status", str(campaign)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["unaccepted_task_keys"] == ["one"]
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    assert "unaccepted tasks" in capsys.readouterr().out

    with pytest.raises(SystemExit) as plan_help_exit:
        main(["plan", "--help"])
    assert plan_help_exit.value.code == 0
    plan_help = capsys.readouterr().out
    assert "--completed" in plan_help
    assert "--retry" in plan_help


def test_cli_plan_output_is_owner_only_and_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tasks, resources, target = write_inputs(tmp_path)
    output = tmp_path / "plans" / "PLAN.json"
    arguments = plan_arguments(tasks, resources, target, tmp_path / "campaign", output)
    original_write = Path.write_bytes
    modes_at_write: list[int] = []

    def observe_stage_mode(path: Path, data: bytes) -> int:
        if path.name.startswith(".servatus-file-stage-"):
            modes_at_write.append(stat.S_IMODE(path.stat().st_mode))
        return original_write(path, data)

    monkeypatch.setattr(Path, "write_bytes", observe_stage_mode)

    previous_umask = os.umask(0)
    try:
        assert main(arguments) == 0
    finally:
        os.umask(previous_umask)
    original = output.read_bytes()
    assert modes_at_write == [0o600]
    assert output.stat().st_mode & 0o777 == 0o600
    capsys.readouterr()

    with pytest.raises(SystemExit) as duplicate:
        main(arguments)
    assert duplicate.value.code == 2
    assert "destination already exists" in capsys.readouterr().err
    assert output.read_bytes() == original


def test_cli_plan_exposes_completed_and_retry_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks_path, resources, target = write_inputs(tmp_path)
    stdin = tmp_path / "stdin.bin"
    tasks_path.write_text(
        "".join(
            json.dumps(
                {"key": f"task-{index}", "args": ["run", str(index)], "stdin_file": stdin.name}
            )
            + "\n"
            for index in range(4)
        )
    )
    campaign = tmp_path / "campaign"
    original = tmp_path / "PLAN.json"
    main(plan_arguments(tasks_path, resources, target, campaign, original))
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *args, **kwargs: _slurm.Result(0, b"700;alpha\n", b""),
    )
    main(["submit", str(campaign), str(original)])
    capsys.readouterr()

    retry_plan = tmp_path / "RETRY.json"
    assert (
        main(
            plan_arguments(
                tasks_path,
                resources,
                target,
                campaign,
                retry_plan,
                "--completed",
                "task-2",
                "--completed",
                "task-3",
                "--retry",
                "task-0",
            )
        )
        == 0
    )
    document = json.loads(retry_plan.read_text())
    assert document["completed"] == ["task-2", "task-3"]
    assert document["retry"] == ["task-0"]
    assert [allocation["task_keys"] for allocation in document["allocations"]] == [["task-0"]]


def test_cli_reconciles_with_campaign_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks, resources, target = write_inputs(tmp_path)
    campaign = tmp_path / "campaign"
    output = tmp_path / "PLAN.json"
    main(plan_arguments(tasks, resources, target, campaign, output))
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *args, **kwargs: _slurm.Result(1, b"", b"lost reply"),
    )
    capsys.readouterr()
    with pytest.raises(SystemExit):
        main(["submit", str(campaign), str(output)])
    capsys.readouterr()
    main(["status", str(campaign)])
    allocation_id = json.loads(capsys.readouterr().out)["ambiguous_allocation_ids"][0]
    queried: list[SlurmTarget] = []

    def query(target_value: SlurmTarget, **kwargs: object) -> _slurm.IdentityMatch:
        queried.append(target_value)
        return _slurm.IdentityMatch(701, "alpha")

    monkeypatch.setattr(_slurm, "query_identity", query)

    assert main(["reconcile", str(campaign), allocation_id]) == 0
    assert queried == [SlurmTarget.from_toml(target)]
    assert json.loads(capsys.readouterr().out)["job_id"] == 701


def test_cli_validate_and_submit_exact_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks, resources, target = write_inputs(tmp_path)
    campaign = tmp_path / "campaign"
    output = tmp_path / "PLAN.json"
    main(plan_arguments(tasks, resources, target, campaign, output))
    capsys.readouterr()
    calls: list[tuple[str, ...]] = []

    def fake(target_value: object, argv: tuple[str, ...], script: bytes) -> _slurm.Result:
        calls.append(argv)
        if argv[-1] == "--test-only":
            return _slurm.Result(0, b"valid now\n", b"")
        return _slurm.Result(0, b"700;alpha\n", b"")

    monkeypatch.setattr(_slurm, "_run_ssh", fake)
    assert main(["validate", str(campaign), str(output)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["time_specific"] is True
    assert len(validation["results"]) == 1
    result = validation["results"][0]
    assert result["shape"] == {
        "task_count": 1,
        "cpus": 2,
        "memory_mib": 1024,
        "gpus": 0,
        "time_limit": "00:10:00",
    }
    assert len(result["shape_digest"]) == 64
    assert len(result["script_digest"]) == 64
    assert result["controller_stdout"] == "valid now"
    assert main(["submit", str(campaign), str(output)]) == 0
    assert any(argv[-1] == "--test-only" for argv in calls)
    assert any(argv[-1] != "--test-only" for argv in calls)


def test_cli_sensitive_script_diagnostic_is_explicit_and_plan_stays_redacted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks, resources, target = write_inputs(tmp_path)
    output = tmp_path / "PLAN.json"
    assert (
        main(
            plan_arguments(
                tasks,
                resources,
                target,
                tmp_path / "campaign",
                output,
                "--show-scripts",
            )
        )
        == 0
    )
    captured = capsys.readouterr()
    diagnostic = json.loads(captured.out)
    assert diagnostic["sensitive"] is True
    assert "expose task arguments and payloads" in captured.err
    script = diagnostic["scripts"][0]["script"]
    assert "run" in script
    assert base64.b64encode(b"opaque\x00payload").decode() in script
    persisted = output.read_text()
    assert "opaque" not in persisted
    assert "payload" not in persisted
    assert base64.b64encode(b"opaque\x00payload").decode() not in persisted


def test_cli_malformed_plan_is_concise_nonzero_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks, resources, target = write_inputs(tmp_path)
    campaign = tmp_path / "campaign"
    output = tmp_path / "PLAN.json"
    main(plan_arguments(tasks, resources, target, campaign, output))
    document = json.loads(output.read_text())
    document["schema_version"] = True
    output.write_text(json.dumps(document))
    capsys.readouterr()
    with pytest.raises(SystemExit) as exit_status:
        main(["submit", str(campaign), str(output)])
    assert exit_status.value.code == 2
    error = capsys.readouterr().err
    assert "plan document schema is unsupported" in error
    assert "Traceback" not in error
