from __future__ import annotations

import json
from pathlib import Path

import pytest

from servatus import _slurm
from servatus.cli import main


def write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    stdin = tmp_path / "stdin.bin"
    stdin.write_bytes(b"opaque\x00payload")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"key": "one", "args": ["run"], "stdin_file": str(stdin)}) + "\n")
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


def test_cli_plan_status_and_help(tmp_path: Path, capsys: object) -> None:
    tasks, resources, target = write_inputs(tmp_path)
    campaign = tmp_path / "campaign"
    output = tmp_path / "PLAN.json"
    assert (
        main(
            [
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
            ]
        )
        == 0
    )
    document = output.read_text()
    assert "opaque" not in document
    assert "payload" not in document
    assert '"one"' in document
    assert main(["status", str(campaign)]) == 0
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0


def test_cli_validate_and_submit_exact_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks, resources, target = write_inputs(tmp_path)
    campaign = tmp_path / "campaign"
    output = tmp_path / "PLAN.json"
    main(
        [
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
        ]
    )
    calls: list[tuple[str, ...]] = []

    def fake(target_value: object, argv: tuple[str, ...], script: bytes) -> _slurm.Result:
        calls.append(argv)
        if argv[-1] == "--test-only":
            return _slurm.Result(0, b"valid now\n", b"")
        return _slurm.Result(0, b"700;alpha\n", b"")

    monkeypatch.setattr(_slurm, "_run_ssh", fake)
    assert main(["validate", str(campaign), str(output)]) == 0
    assert main(["submit", str(campaign), str(output)]) == 0
    assert any(argv[-1] == "--test-only" for argv in calls)
    assert any(argv[-1] != "--test-only" for argv in calls)
