"""Live end-to-end check of Servatus against a real Slurm cluster. Never run by CI or pytest.

    uv run python scripts/cluster_e2e.py HOST WORK_DIR --partition NAME [--account A] [--qos Q]
        [--gpu-partition NAME --gpu-gres GRES] [--image-partition NAME]

It drives small sweeps through the ``servatus`` CLI over SSH, one Task per allocation, each with
1 CPU, 1 GiB at most, and a 10-minute limit. ``servatus doctor`` must pass every profile first.

- Direct launcher, ``/bin/sh`` Tasks: ``ok`` exits 0, ``fail`` exits 3, and ``long`` sleeps until
  ``servatus cancel`` stops it. After Slurm purges the finished jobs, ``status --json`` must show
  SUCCEEDED, FAILED (exit 3), and STOPPED, and be quiescent, and ``plan --retry-failed`` must select
  exactly ``fail`` and ``long``.
- ``--gpu-partition``: one one-GPU Task runs ``nvidia-smi -L`` through the direct launcher and must
  list exactly one GPU.
- ``--image-partition``: a compute node pulls a tiny image, then one Task runs in it through the
  Apptainer launcher. A failed pull fails the check, so leave this off where nodes have no network.
- Both: one one-GPU Task runs ``nvidia-smi -L`` inside the image through ``--nv``. It must list
  exactly one GPU and see Slurm's step-local ``CUDA_VISIBLE_DEVICES``.
- Budgeted chains: ``plan --budget 00:09:00`` on 3-minute links with a 90-second pre-limit
  signal queues two chains of three links. On link 1, ``yield`` traps ``SIGUSR1`` and exits 75,
  and ``die`` dies of it. Each must continue on link 2, which exits 0, after which Slurm must
  delete link 3 unstarted. This proves on the cluster what the test suite can only emulate:
  ``--signal=B:USR1`` reaching the batch shell (its limit marker), ``srun`` forwarding the relayed
  signal to the step (``yield`` logs it), ``afternotok`` with ``--kill-on-invalid-dep=yes``
  pruning the chain, the exact-name handoff lookup across nodes, and ``date`` on compute nodes
  (link 2 learns a positive ``SERVATUS_RUN_TIME``).
- Link 2 of each chain (direct launcher) and, with ``--image-partition``, the Apptainer step
  under ``--cleanenv`` must learn a positive ``SERVATUS_JOB_END_TIME`` from Slurm's
  ``SLURM_JOB_END_TIME``.

Everything it writes on the cluster lives in a new ``WORK_DIR/run-<timestamp>`` directory. It
submits at most thirteen small jobs, at most three running at a time, and cancels whatever it
leaves behind. It prints one JSON summary line last.
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
REPORT = (
    'echo "link $SERVATUS_LINK of $SERVATUS_LINKS ran $SERVATUS_RUN_TIME of $SERVATUS_BUDGET'
    ' ends $SERVATUS_JOB_END_TIME"'
)
CHAIN_TASKS = {
    "yield": (
        f'{REPORT}; if [ "$SERVATUS_LINK" = 1 ]; then '
        'trap "echo got USR1; exit 75" USR1; sleep 600 & wait $!; fi; exit 0'
    ),
    "die": f'{REPORT}; if [ "$SERVATUS_LINK" = 1 ]; then sleep 600; fi; exit 0',
}
DIED_OF_USR1 = 138
# glibc, not Alpine's musl: --nv binds the host's nvidia-smi, which needs a glibc loader.
IMAGE_SOURCE = "docker://debian:12-slim"
PULL_JOB = "servatus-e2e-pull"
PULL_SECONDS = 600
IMAGE_TASK = (
    'cat /etc/debian_version; printf "%s %s %s\\n" "$SERVATUS_TASK_KEY" "$PWD" '
    '"$SERVATUS_JOB_END_TIME"'
)
IMAGE_GPU_TASK = (
    'cat /etc/debian_version && nvidia-smi -L && printf "%s %s\\n" "$SERVATUS_TASK_KEY" '
    '"$CUDA_VISIBLE_DEVICES"'
)


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


EXIT_UNAVAILABLE = 75


def status(root: Path, state: str = "state") -> dict[str, Any] | None:
    """The status document, or None when the cluster was briefly unavailable (exit 75)."""
    completed = servatus(root, "status", state, "--json", check=False)
    if completed.returncode == EXIT_UNAVAILABLE:
        log(f"status unavailable, retrying: {completed.stderr.strip()}")
        return None
    if completed.returncode != 0:
        raise SystemExit(f"servatus status exited {completed.returncode}:\n{completed.stderr}")
    return cast(dict[str, Any], json.loads(completed.stdout))


def executions(document: dict[str, Any]) -> dict[str, str | None]:
    return {task["key"]: task["execution"] for task in document["tasks"]}


def wait_for(
    root: Path,
    what: str,
    deadline: float,
    done: Callable[[dict[str, Any]], bool],
    state: str = "state",
) -> dict[str, Any]:
    while True:
        document = status(root, state)
        if document is not None and done(document):
            return document
        shown = "unavailable" if document is None else executions(document)
        if time.monotonic() > deadline:
            raise SystemExit(f"timed out waiting for {what}: {shown}")
        log(f"waiting for {what}: {shown}")
        time.sleep(POLL_SECONDS)


def configuration(arguments: argparse.Namespace, run: PurePosixPath, image: bool) -> str:
    optional = "".join(
        f"{name} = {json.dumps(value)}\n"
        for name, value in (("account", arguments.account), ("qos", arguments.qos))
        if value is not None
    )
    document = (
        'default_profile = "e2e"\n\n'
        "[target]\n"
        f"host = {json.dumps(arguments.host)}\n"
        f"slurm_bin = {json.dumps(arguments.slurm_bin)}\n"
        f'work_root = "{run}/work"\n'
        f'log_root = "{run}/logs"\n'
        f"{optional}"
        "max_tasks_per_allocation = 1\n"
        "max_cpus_per_allocation = 1\n"
        "max_memory_mib_per_allocation = 1024\n"
        'max_time_limit = "00:15:00"\n'
        "\n[resources]\n"
        "cpus = 1\n"
        "memory_mib = 256\n"
        'time_limit = "00:10:00"\n'
        "\n[profiles.e2e.target]\n"
        f"partitions = [{json.dumps(arguments.partition)}]\n"
        "\n[profiles.chain.target]\n"
        f"partitions = [{json.dumps(arguments.partition)}]\n"
        "\n[profiles.chain.resources]\n"
        'time_limit = "00:03:00"\n'
        'signal_before_end = "00:01:30"\n'
    )
    if arguments.gpu_partition is not None:
        document += (
            "\n[profiles.gpu.target]\n"
            f"partitions = [{json.dumps(arguments.gpu_partition)}]\n"
            f"gpu_gres = {json.dumps(arguments.gpu_gres)}\n"
            "max_gpus_per_allocation = 1\n"
            "\n[profiles.gpu.resources]\n"
            "gpus = 1\n"
            "memory_mib = 1024\n"
        )
    if image:
        document += (
            "\n[profiles.image.target]\n"
            f"partitions = [{json.dumps(arguments.image_partition)}]\n"
            f"apptainer = {json.dumps(arguments.apptainer)}\n"
            f'image = "{run}/image.sif"\n'
        )
    if image and arguments.gpu_partition is not None:
        document += (
            "\n[profiles.image-gpu.target]\n"
            f"partitions = [{json.dumps(arguments.gpu_partition)}]\n"
            f"gpu_gres = {json.dumps(arguments.gpu_gres)}\n"
            "max_gpus_per_allocation = 1\n"
            f"apptainer = {json.dumps(arguments.apptainer)}\n"
            f'image = "{run}/image.sif"\n'
            "\n[profiles.image-gpu.resources]\n"
            "gpus = 1\n"
            "memory_mib = 1024\n"
        )
    return document


def diagnose(root: Path, profile: str) -> dict[str, Any]:
    """``servatus doctor`` for ``profile``, which must pass every check."""
    document = cast(
        dict[str, Any], json.loads(servatus(root, "doctor", "--profile", profile, "--json").stdout)
    )
    log(f"doctor {profile}: {document['verdict']}")
    return document


def one_task(root: Path, profile: str, argv: list[str], deadline: float) -> tuple[int, str]:
    """Run one Task under ``profile`` to success; its job number and step log."""
    (root / f"{profile}.jsonl").write_text(json.dumps({"key": profile, "args": argv}) + "\n")
    servatus(root, "create", profile, f"{profile}.jsonl")
    plan = f"{profile}.plan.json"
    servatus(root, "plan", profile, "--profile", profile, "--output", plan)
    servatus(root, "validate", profile, plan)
    submitted = json.loads(servatus(root, "submit", profile, plan, "--json").stdout)
    job = submitted["receipts"][0]["job"]
    log(f"submitted {profile} as job {job}")
    final = wait_for(
        root,
        f"{profile} to finish",
        deadline,
        lambda d: executions(d)[profile] in ("SUCCEEDED", "FAILED", "STOPPED", "LOST"),
        profile,
    )
    output = root / f"{profile}.log"
    while True:
        read = servatus(
            root, "logs", profile, "--task", profile, "--output", str(output), check=False
        )
        if read.returncode == 0:
            break
        if time.monotonic() > deadline:
            raise SystemExit(f"cannot read the {profile} log:\n{read.stderr}")
        log(f"waiting for the {profile} log: {read.stderr.strip()}")
        time.sleep(POLL_SECONDS)
    text = output.read_text()
    assert executions(final)[profile] == "SUCCEEDED", (final, text)
    return job, text


def one_gpu(text: str) -> list[str]:
    gpus = [line for line in text.splitlines() if line.startswith("GPU ")]
    assert len(gpus) == 1, text
    return gpus


def gpu_check(root: Path, deadline: float) -> dict[str, Any]:
    job, text = one_task(root, "gpu", ["/usr/bin/nvidia-smi", "-L"], deadline)
    return {"job": job, "gpus": one_gpu(text)}


def pull_image(arguments: argparse.Namespace, run: PurePosixPath) -> None:
    """Pull the tiny image on a compute node."""
    pull = (
        f"{arguments.slurm_bin}/srun",
        f"--partition={arguments.image_partition}",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--mem=2G",
        "--time=00:15:00",
        f"--chdir={run}",
        f"--job-name={PULL_JOB}",
        "/usr/bin/env",
        f"APPTAINER_TMPDIR={run}/tmp",
        arguments.apptainer,
        "pull",
        "--disable-cache",
        "image.sif",
        IMAGE_SOURCE,
    )
    log(f"pulling {IMAGE_SOURCE} on a {arguments.image_partition} node")
    try:
        completed = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", arguments.host, shlex.join(pull)],
            capture_output=True,
            text=True,
            check=False,
            timeout=PULL_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise SystemExit(f"the image pull took over {PULL_SECONDS} s") from None
    if completed.returncode != 0:
        tail = completed.stderr.strip().splitlines()[-1:]
        raise SystemExit(f"the image pull exited {completed.returncode}: {' '.join(tail)}")


def image_check(root: Path, run: PurePosixPath, deadline: float) -> dict[str, Any]:
    job, text = one_task(root, "image", ["/bin/sh", "-c", IMAGE_TASK], deadline)
    release, identity = text.splitlines()[-2:]
    identity, end_time = identity.rsplit(" ", 1)
    assert release.startswith("12."), text
    assert identity == f"image {run}/work", text
    assert positive(end_time), f"the Apptainer step learned no job end time: {text}"
    return {"job": job, "debian": release}


def image_gpu_check(root: Path, deadline: float) -> dict[str, Any]:
    job, text = one_task(root, "image-gpu", ["/bin/sh", "-c", IMAGE_GPU_TASK], deadline)
    lines = text.splitlines()
    key, visible = lines[-1].split(" ", 1)
    assert lines[-3].startswith("12.") and key == "image-gpu" and visible, text
    return {"job": job, "debian": lines[-3], "gpus": one_gpu(text), "visible": visible}


def positive(text: str) -> bool:
    return text.isdigit() and int(text) > 0


def seconds(duration: str) -> int:
    days, _, clock = duration.rpartition("-")
    hours, minutes, rest = (int(part) for part in clock.split(":"))
    return int(days or 0) * 86_400 + hours * 3_600 + minutes * 60 + rest


def step_log(root: Path, allocation_id: str, key: str, deadline: float) -> str:
    output = root / f"chain-{allocation_id}-{key}.log"
    while True:
        read = servatus(
            root,
            "logs",
            "chain",
            "--allocation",
            allocation_id,
            "--task",
            key,
            "--output",
            str(output),
            check=False,
        )
        if read.returncode == 0:
            return output.read_text()
        if time.monotonic() > deadline:
            raise SystemExit(f"cannot read the {key} log of {allocation_id}:\n{read.stderr}")
        time.sleep(POLL_SECONDS)


def chain_check(root: Path, deadline: float) -> dict[str, Any]:
    """Two budgeted chains of three links: each continues once, ends on link 2, and Slurm deletes
    link 3 before it starts."""
    diagnose(root, "chain")
    (root / "chain.jsonl").write_text(
        "".join(
            json.dumps({"key": key, "args": ["/bin/sh", "-c", command]}) + "\n"
            for key, command in CHAIN_TASKS.items()
        )
    )
    servatus(root, "create", "chain", "chain.jsonl")
    planned = json.loads(
        servatus(
            root,
            "plan",
            "chain",
            "--profile",
            "chain",
            "--budget",
            "00:09:00",
            "--output",
            "chain.plan.json",
            "--json",
        ).stdout
    )
    links = [(item["link"]["index"], item["time_limit"]) for item in planned["allocations"]]
    assert links == [(index, "00:03:00") for index in (1, 1, 2, 2, 3, 3)], planned
    servatus(root, "validate", "chain", "chain.plan.json")
    submitted = json.loads(servatus(root, "submit", "chain", "chain.plan.json", "--json").stdout)
    assert len(submitted["receipts"]) == 6, submitted
    log(f"submitted chain jobs {[receipt['job'] for receipt in submitted['receipts']]}")
    final = wait_for(
        root,
        "the chains to end and Slurm to purge them",
        deadline,
        lambda d: d["quiescent"],
        "chain",
    )
    tasks = {task["key"]: task for task in final["tasks"]}
    links_of: dict[str, dict[int, dict[str, Any]]] = {}
    for attempt in final["attempts"]:
        links_of.setdefault(attempt["task_keys"][0], {})[attempt["link"]["index"]] = attempt
    summary: dict[str, Any] = {}
    for key, code in (("yield", 75), ("die", DIED_OF_USR1)):
        task, chain = tasks[key], links_of[key]
        assert (task["execution"], task["exit_code"]) == ("SUCCEEDED", 0), task
        assert (task["link"]["index"], task["link"]["count"]) == (2, 3), task
        assert task["run_time"]["complete"], task
        first, last = chain[1]["observation"], chain[3]["observation"]
        assert first["limit_at"] is not None, f"{key}: the batch shell never got SIGUSR1"
        assert first["ends"][0]["exit_code"] == code, first
        assert last["started_at"] is None, f"{key}: Slurm started link 3"
        second = step_log(root, chain[2]["allocation_id"], key, deadline)
        line = second.strip().splitlines()[0].split()
        assert line[:4] == ["link", "2", "of", "3"] and line[7] == "540", second
        assert int(line[5]) > 0, f"{key}: link 2 learned no run time: {second}"
        assert line[8] == "ends" and positive(line[9]), f"{key}: no job end time: {second}"
        if key == "yield":
            assert "got USR1" in step_log(root, chain[1]["allocation_id"], key, deadline)
        summary[key] = {
            "run_time": seconds(task["run_time"]["total"]),
            "link_2_learned": int(line[5]),
        }
    return summary


def sweep(arguments: argparse.Namespace, root: Path, run: PurePosixPath) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + arguments.timeout
    (root / "SERVATUS.toml").write_text(configuration(arguments, run, False))
    doctor = diagnose(root, "e2e")
    image = arguments.image_partition is not None
    if image:
        pull_image(arguments, run)
    (root / "SERVATUS.toml").write_text(configuration(arguments, run, image))
    (root / "tasks.jsonl").write_text(
        "".join(
            json.dumps({"key": key, "args": ["/bin/sh", "-c", command]}) + "\n"
            for key, command in TASKS.items()
        )
    )
    image_gpu = image and arguments.gpu_partition is not None
    for profile, wanted in (
        ("gpu", arguments.gpu_partition is not None),
        ("image", image),
        ("image-gpu", image_gpu),
    ):
        if wanted:
            diagnose(root, profile)
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
    assert observation["started_at"] and observation["stopped_at"], long_attempt

    retry = json.loads(servatus(root, "plan", "state", "--retry-failed", "--json").stdout)
    assert retry["selected"] == ["fail", "long"], retry
    assert retry["held"] == {"ok": "SUBMITTED"}, retry
    summary: dict[str, Any] = {
        "host": arguments.host,
        "run": str(run),
        "slurm": doctor["checks"][0]["detail"],
        "jobs": jobs,
        "executions": {key: list(value) for key, value in outcome.items()},
        "retry_failed": retry["selected"],
    }
    if arguments.gpu_partition is not None:
        summary["gpu"] = gpu_check(root, deadline)
    if image:
        summary["apptainer"] = image_check(root, run, deadline)
        if image_gpu:
            summary["apptainer_gpu"] = image_gpu_check(root, deadline)
    summary["chains"] = chain_check(root, deadline)
    summary["seconds"] = round(time.monotonic() - started)
    return summary


def cancel_leftovers(arguments: argparse.Namespace, root: Path) -> None:
    """Stop what this run may have left. Each Task is cancelled on its own, because one without an
    accepted allocation refuses a whole cancel."""
    if arguments.image_partition is not None:
        pull = f"{arguments.slurm_bin}/scancel --me --name={PULL_JOB}"
        subprocess.run(["ssh", "-o", "BatchMode=yes", arguments.host, pull], check=False)
    for state, keys in (
        ("state", tuple(TASKS)),
        ("gpu", ("gpu",)),
        ("image", ("image",)),
        ("image-gpu", ("image-gpu",)),
        ("chain", tuple(CHAIN_TASKS)),
    ):
        if (root / state).exists():
            for key in keys:
                cancelled = servatus(root, "cancel", state, "--task", key, check=False)
                if cancelled.returncode != 0:
                    log(f"could not cancel {key}: {cancelled.stderr.strip()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live end-to-end check of Servatus on Slurm.")
    parser.add_argument("host", help="SSH destination (ssh_config alias)")
    parser.add_argument("work_dir", help="absolute directory on the cluster for this check")
    parser.add_argument("--partition", required=True)
    parser.add_argument("--account")
    parser.add_argument("--qos")
    parser.add_argument("--slurm-bin", default="/usr/bin")
    parser.add_argument("--gpu-partition", help="also run a one-GPU Task in this partition")
    parser.add_argument("--gpu-gres", default="gpu", help="count-free GRES for --gpu-partition")
    parser.add_argument("--image-partition", help="also pull an image and run a Task in it here")
    parser.add_argument("--apptainer", default="/usr/bin/apptainer")
    parser.add_argument("--timeout", type=int, default=2700, help="seconds for the whole sweep")
    arguments = parser.parse_args()
    work_dir = PurePosixPath(arguments.work_dir)
    if not work_dir.is_absolute():
        raise SystemExit("WORK_DIR must be absolute")
    run = work_dir / f"run-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    directories = " ".join(shlex.quote(str(run / name)) for name in ("work", "logs", "tmp"))
    remote = f"mkdir -p -m 700 {directories}"
    subprocess.run(["ssh", "-o", "BatchMode=yes", arguments.host, remote], check=True)
    log(f"run directory {arguments.host}:{run}")
    with tempfile.TemporaryDirectory(prefix="servatus-e2e-") as directory:
        root = Path(directory)
        try:
            summary = sweep(arguments, root, run)
        finally:
            cancel_leftovers(arguments, root)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
