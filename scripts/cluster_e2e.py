"""Live end-to-end check of Servatus against a real Slurm cluster. Never run by CI or pytest.

    uv run python scripts/cluster_e2e.py HOST WORK_DIR --partition NAME [--account A] [--qos Q]

It drives one small sweep through the ``servatus`` CLI over SSH, using the direct launcher and
``/bin/sh`` Tasks with 1 CPU, 256 MiB, and a 10-minute limit each, one Task per allocation:

- ``ok`` exits 0, ``fail`` exits 3, and ``long`` sleeps until ``servatus cancel`` stops it;
- after Slurm purges the finished jobs, ``status --json`` must show SUCCEEDED, FAILED (exit 3),
  and STOPPED, and be quiescent;
- ``plan --retry-failed`` must select exactly ``fail`` and ``long``.

Everything it writes on the cluster lives in a new ``WORK_DIR/run-<timestamp>`` directory. It
submits three jobs and cancels whatever it leaves behind. It prints one JSON summary line last.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast

TASKS = {
    "ok": "exit 0",
    "fail": "exit 3",
    "long": "sleep 900",
}
POLL_SECONDS = 20


def log(message: str) -> None:
    stamp = datetime.now(UTC).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


def servatus(root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [sys.executable, "-m", "servatus", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and completed.returncode != 0:
        raise SystemExit(
            f"servatus {' '.join(arguments)} exited {completed.returncode}:\n"
            f"{completed.stdout}{completed.stderr}"
        )
    return completed


def status(root: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(servatus(root, "status", "state", "--json").stdout))


def executions(document: dict[str, Any]) -> dict[str, str | None]:
    return {task["key"]: task["execution"] for task in document["tasks"]}


def wait_for(
    root: Path, what: str, deadline: float, done: Callable[[dict[str, Any]], bool]
) -> dict[str, Any]:
    while True:
        document = status(root)
        if done(document):
            return document
        if time.monotonic() > deadline:
            raise SystemExit(f"timed out waiting for {what}: {executions(document)}")
        log(f"waiting for {what}: {executions(document)}")
        time.sleep(POLL_SECONDS)


def configuration(arguments: argparse.Namespace, run: PurePosixPath) -> str:
    optional = "".join(
        f"{name} = {json.dumps(value)}\n"
        for name, value in (("account", arguments.account), ("qos", arguments.qos))
        if value is not None
    )
    return (
        "[profiles.e2e.target]\n"
        f"host = {json.dumps(arguments.host)}\n"
        f"slurm_bin = {json.dumps(arguments.slurm_bin)}\n"
        f'work_root = "{run}/work"\n'
        f'log_root = "{run}/logs"\n'
        f"partitions = [{json.dumps(arguments.partition)}]\n"
        f"{optional}"
        "max_tasks_per_allocation = 1\n"
        "max_cpus_per_allocation = 1\n"
        "max_memory_mib_per_allocation = 256\n"
        'max_time_limit = "00:15:00"\n'
        "\n[profiles.e2e.resources]\n"
        "cpus = 1\n"
        "memory_mib = 256\n"
        'time_limit = "00:10:00"\n'
    )


def sweep(arguments: argparse.Namespace, root: Path, run: PurePosixPath) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + arguments.timeout
    (root / "SERVATUS.toml").write_text(configuration(arguments, run))
    (root / "tasks.jsonl").write_text(
        "".join(
            json.dumps({"key": key, "args": ["/bin/sh", "-c", command]}) + "\n"
            for key, command in TASKS.items()
        )
    )
    doctor = json.loads(servatus(root, "doctor", "--json").stdout)
    servatus(root, "create", "state", "tasks.jsonl")
    servatus(root, "plan", "state", "--output", "plan.json")
    servatus(root, "validate", "state", "plan.json")
    submitted = json.loads(servatus(root, "submit", "state", "plan.json", "--json").stdout)
    jobs = {receipt["task_keys"][0]: receipt["job"] for receipt in submitted["receipts"]}
    assert sorted(jobs) == sorted(TASKS), submitted
    log(f"submitted jobs {jobs}")

    wait_for(root, "long to run", deadline, lambda d: executions(d)["long"] == "RUNNING")
    cancelled = json.loads(servatus(root, "cancel", "state", "--task", "long", "--json").stdout)
    assert [item["job"] for item in cancelled["cancelled"]] == [jobs["long"]], cancelled
    log(f"cancelled job {jobs['long']}")

    final = wait_for(root, "Slurm to purge every job", deadline, lambda d: d["quiescent"])
    tasks = {task["key"]: task for task in final["tasks"]}
    outcome = {key: (task["execution"], task["exit_code"]) for key, task in tasks.items()}
    expected = {"ok": ("SUCCEEDED", 0), "fail": ("FAILED", 3), "long": ("STOPPED", None)}
    assert outcome == expected, outcome
    long_attempt = next(item for item in final["attempts"] if item["job"] == jobs["long"])
    observation = long_attempt["observation"]
    assert observation["started"] and observation["stopped"], long_attempt

    retry = json.loads(servatus(root, "plan", "state", "--retry-failed", "--json").stdout)
    assert retry["selected"] == ["fail", "long"], retry
    assert retry["held"] == {"ok": "SUBMITTED"}, retry
    return {
        "host": arguments.host,
        "run": str(run),
        "sbatch": doctor["sbatch_version"],
        "jobs": jobs,
        "executions": {key: list(value) for key, value in outcome.items()},
        "retry_failed": retry["selected"],
        "seconds": round(time.monotonic() - started),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Live end-to-end check of Servatus on Slurm.")
    parser.add_argument("host", help="SSH destination (ssh_config alias)")
    parser.add_argument("work_dir", help="absolute directory on the cluster for this check")
    parser.add_argument("--partition", required=True)
    parser.add_argument("--account")
    parser.add_argument("--qos")
    parser.add_argument("--slurm-bin", default="/usr/bin")
    parser.add_argument("--timeout", type=int, default=1500, help="seconds for the whole sweep")
    arguments = parser.parse_args()
    work_dir = PurePosixPath(arguments.work_dir)
    if not work_dir.is_absolute():
        raise SystemExit("WORK_DIR must be absolute")
    run = work_dir / f"run-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    remote = f"mkdir -p -m 700 {shlex.quote(str(run / 'work'))} {shlex.quote(str(run / 'logs'))}"
    subprocess.run(["ssh", "-o", "BatchMode=yes", arguments.host, remote], check=True)
    log(f"run directory {arguments.host}:{run}")
    with tempfile.TemporaryDirectory(prefix="servatus-e2e-") as directory:
        root = Path(directory)
        try:
            summary = sweep(arguments, root, run)
        finally:
            if (root / "state").exists():
                keys = [option for key in TASKS for option in ("--task", key)]
                servatus(root, "cancel", "state", *keys, check=False)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
