"""Installed-artifact smoke test.

Run against a built wheel in an isolated environment, from any directory:

    uv run --isolated --no-project --with dist/servatus-*.whl \
        python .github/scripts/smoke.py dist/servatus-*.whl

It never contacts SSH, Slurm, or Apptainer; everything happens in a temporary directory.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from importlib.metadata import version
from pathlib import Path

PUBLIC_NAMES = (
    "Apptainer",
    "Campaign",
    "Draft",
    "Profile",
    "Publication",
    "Resources",
    "Retry",
    "ServatusError",
    "Target",
    "Task",
    "Workspace",
    "publish",
    "publish_file",
)


def run(*argv: str, cwd: Path) -> str:
    completed = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"{' '.join(argv)} exited {completed.returncode}:\n{completed.stderr}")
    return completed.stdout


def check_wheel(wheel: Path) -> None:
    names = zipfile.ZipFile(wheel).namelist()
    assert "servatus/py.typed" in names, "py.typed is missing from the wheel"
    assert not any(name.startswith("tests/") for name in names), "tests leaked into the wheel"


def check_imports() -> None:
    import servatus.testing  # also binds servatus

    missing = [name for name in PUBLIC_NAMES if not hasattr(servatus, name)]
    assert not missing, f"missing public names: {missing}"
    assert hasattr(servatus.testing, "FakeScheduler")
    assert servatus.__version__ == version("servatus"), servatus.__version__
    # `uv run --with` installs into a cached layer outside sys.prefix, so check the source instead.
    imported = Path(servatus.__file__).resolve()
    repository = Path(__file__).resolve().parents[2]
    assert "site-packages" in imported.parts and not imported.is_relative_to(repository), (
        f"imported servatus from {imported}, not the installed wheel"
    )


def check_cli(root: Path) -> None:
    executable = shutil.which("servatus")
    assert executable is not None, "the servatus console script is not on PATH"
    assert version("servatus") in run(executable, "--version", cwd=root)
    run(sys.executable, "-m", "servatus", "--help", cwd=root)

    tasks = root / "tasks.jsonl"
    tasks.write_text(
        '{"key": "alpha", "args": ["/bin/true"]}\n'
        '{"key": "beta", "args": ["/bin/true"], "env": {"MODE": "smoke"}}\n'
    )
    run(executable, "create", "state", "tasks.jsonl", cwd=root)
    document = json.loads(run(executable, "status", "state", "--offline", "--json", cwd=root))
    assert document["format"] == "servatus.status/1", document
    assert [task["key"] for task in document["tasks"]] == ["alpha", "beta"], document


def check_campaign(root: Path) -> None:
    from servatus import Campaign, Profile, Resources, Retry, Target, Task
    from servatus.testing import FakeScheduler

    fake = FakeScheduler()
    target = Target(
        slurm_bin="/opt/slurm/bin",
        work_root="/work",
        log_root="/logs",
        partitions=["cpu"],
        max_tasks_per_allocation=2,
        max_cpus_per_allocation=4,
        max_memory_mib_per_allocation=4096,
        max_time_limit="01:00:00",
    )
    profile = Profile("smoke", target, Resources(cpus=1, memory_mib=512, time_limit="00:05:00"))
    tasks = [Task(f"t{index}", ["/bin/echo", str(index)]) for index in range(3)]
    campaign = Campaign.create(root / "campaign", tasks, connect=fake)
    plan = campaign.load_plan(campaign.plan(profile).to_json())
    assert [len(item.task_keys) for item in plan.allocations] == [2, 1], plan
    assert all(check.accepted for check in campaign.validate(plan))
    result = campaign.submit(plan)
    assert result.complete and len(result.receipts) == 2, result
    fake.start(result.receipts[0].job)
    fake.finish(result.receipts[0].job, "FAILED", exit_code="1:0")
    assert campaign.status().counts()["failed"] == 2
    assert campaign.plan(profile, retry=Retry.FAILED).decision.selected == ("t0", "t1")


def check_publication(root: Path) -> None:
    from servatus import publish, publish_file
    from servatus.errors import DestinationExists

    def write(path: Path) -> None:
        # A writer that uses temp-file-plus-rename, as many libraries do.
        partial = path.with_name(path.name + ".partial")
        partial.write_bytes(b"payload\n")
        os.replace(partial, path)

    destination = root / "outputs" / "weights.bin"
    destination.parent.mkdir()
    publication = publish_file(destination, write, mode=0o640)
    assert destination.read_bytes() == b"payload\n"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o640
    assert not publication.cleanup_pending
    assert sorted(path.name for path in destination.parent.iterdir()) == ["weights.bin"]
    try:
        publish_file(destination, write)
    except DestinationExists:
        pass
    else:
        raise AssertionError("publish_file overwrote an existing destination")

    source = root / "plots"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / "loss.svg").write_text("<svg/>\n")
    tree = root / "outputs" / "run"
    publish(tree, lambda draft: draft.link_tree(source, "plots"), mode=0o755)
    assert (tree / "plots" / "nested" / "loss.svg").read_text() == "<svg/>\n"
    assert stat.S_IMODE(tree.stat().st_mode) == 0o755


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: smoke.py WHEEL")
    check_wheel(Path(sys.argv[1]).resolve())
    check_imports()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        check_cli(root)
        check_campaign(root)
        check_publication(root)
    print(f"servatus {version('servatus')}: installed-artifact smoke passed")


if __name__ == "__main__":
    main()
