from __future__ import annotations

import base64
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from ._errors import ReconciliationError

if TYPE_CHECKING:
    from ._campaign import ResourceRequest, SlurmTarget, Task


@dataclass(frozen=True, slots=True)
class Result:
    returncode: int
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True, slots=True)
class IdentityMatch:
    job_id: int
    cluster: str | None


def render_script(
    target: SlurmTarget,
    resources: ResourceRequest,
    tasks: tuple[Task, ...],
    allocation_id: str,
) -> bytes:
    lines = ["#!/bin/sh", "set -u", "status=0"]
    srun = target.slurm_bin / "srun"
    for index, task in enumerate(tasks, start=1):
        payload = base64.b64encode(task.stdin).decode("ascii")
        step = [
            str(srun),
            "--exclusive",
            "--exact",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={resources.cpus_per_task}",
            f"--mem={resources.memory_mib_per_task}M",
            f"--chdir={target.work_root}",
            f"--output={target.log_root}/servatus-{allocation_id}-{index}.out",
            f"--error={target.log_root}/servatus-{allocation_id}-{index}.err",
        ]
        if resources.gpus_per_task:
            assert target.gpu_gres is not None
            step.append(f"--gres={target.gpu_gres}:{resources.gpus_per_task}")
        container = [
            str(target.apptainer),
            "run",
            "--cleanenv",
            "--bind",
            f"{target.work_root}:{target.work_root}",
            "--pwd",
            str(target.work_root),
        ]
        if resources.gpus_per_task:
            container.append("--nv")
        container.extend((str(target.image), *task.args))
        lines.append(
            f"printf %s {shlex.quote(payload)} | /usr/bin/base64 -d | "
            f"{shlex.join((*step, *container))} &"
        )
        lines.append(f"pid_{index}=$!")
    for index in range(1, len(tasks) + 1):
        lines.append(f'if ! wait "$pid_{index}"; then status=1; fi')
    lines.extend(('exit "$status"', ""))
    return "\n".join(lines).encode("utf-8")


def sbatch_argv(
    target: SlurmTarget,
    resources: ResourceRequest,
    task_count: int,
    allocation_id: str,
    effective_time_limit: str,
) -> tuple[str, ...]:
    identity = f"servatus-{allocation_id}"
    argv = [
        str(target.slurm_bin / "sbatch"),
        "--parsable",
        "--export=NIL",
        "--nodes=1",
        f"--ntasks={task_count}",
        f"--cpus-per-task={resources.cpus_per_task}",
        f"--mem={task_count * resources.memory_mib_per_task}M",
        f"--time={effective_time_limit}",
        f"--partition={','.join(target.partitions)}",
        f"--chdir={target.work_root}",
        f"--job-name={identity}",
        f"--comment={identity}",
        f"--output={target.log_root}/{identity}.out",
        f"--error={target.log_root}/{identity}.err",
    ]
    if target.account is not None:
        argv.append(f"--account={target.account}")
    if target.qos is not None:
        argv.append(f"--qos={target.qos}")
    if target.constraint is not None:
        argv.append(f"--constraint={target.constraint}")
    if resources.gpus_per_task:
        assert target.gpu_gres is not None
        argv.append(f"--gres={target.gpu_gres}:{task_count * resources.gpus_per_task}")
    return tuple(argv)


def _ssh_environment() -> dict[str, str]:
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
    for name in ("HOME", "LOGNAME", "SSH_AUTH_SOCK", "USER"):
        if value := os.environ.get(name):
            environment[name] = value
    return environment


def _run_ssh(target: SlurmTarget, argv: tuple[str, ...], stdin: bytes) -> Result:
    remote = "/usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C " + shlex.join(argv)
    completed = subprocess.run(
        ["ssh", "-T", "-o", "BatchMode=yes", target.host, remote],
        input=stdin,
        capture_output=True,
        check=False,
        env=_ssh_environment(),
    )
    return Result(completed.returncode, completed.stdout, completed.stderr)


_RECEIPT = re.compile(rb"([1-9][0-9]*)(?:;([A-Za-z0-9][A-Za-z0-9._-]*))?\n?\Z")


def parse_receipt(output: bytes) -> tuple[int, str | None]:
    match = _RECEIPT.fullmatch(output)
    if match is None:
        raise ValueError("sbatch did not return one positive job identity")
    job_id = int(match.group(1))
    cluster_bytes = match.group(2)
    return job_id, None if cluster_bytes is None else cluster_bytes.decode("ascii")


def validate_allocation(target: SlurmTarget, argv: tuple[str, ...], script: bytes) -> Result:
    return _run_ssh(target, (*argv, "--test-only"), script)


def query_identity(
    target: SlurmTarget,
    *,
    job_name: str,
    window_start: str,
    window_end: str,
) -> IdentityMatch:
    squeue = _run_ssh(
        target,
        (
            str(target.slurm_bin / "squeue"),
            "--noheader",
            "--name",
            job_name,
            "--format=%i|%j|%k",
        ),
        b"",
    )
    sacct = _run_ssh(
        target,
        (
            str(target.slurm_bin / "sacct"),
            "--noheader",
            "--parsable2",
            "--allocations",
            "--name",
            job_name,
            "--starttime",
            window_start,
            "--endtime",
            window_end,
            "--format=JobIDRaw,JobName,Comment,Cluster",
        ),
        b"",
    )
    if squeue.returncode != 0 or sacct.returncode != 0:
        raise ReconciliationError("scheduler identity query was unavailable")

    candidates: dict[int, set[str | None]] = {}
    for raw in squeue.stdout.decode("utf-8", "strict").splitlines():
        fields = raw.split("|")
        if len(fields) == 3 and fields[1] == job_name and fields[2] == job_name:
            _add_candidate(candidates, fields[0], None)
    for raw in sacct.stdout.decode("utf-8", "strict").splitlines():
        fields = raw.split("|")
        if len(fields) == 4 and fields[1] == job_name and fields[2] == job_name:
            _add_candidate(candidates, fields[0], fields[3] or None)

    if len(candidates) != 1:
        raise ReconciliationError("scheduler query did not prove one exact allocation identity")
    job_id, clusters = next(iter(candidates.items()))
    known_clusters = {cluster for cluster in clusters if cluster is not None}
    if len(known_clusters) > 1:
        raise ReconciliationError("scheduler query returned conflicting cluster identities")
    return IdentityMatch(job_id, next(iter(known_clusters), None))


def _add_candidate(
    candidates: dict[int, set[str | None]], raw_job_id: str, cluster: str | None
) -> None:
    if not raw_job_id.isascii() or not raw_job_id.isdecimal():
        return
    job_id = int(raw_job_id)
    if job_id <= 0:
        return
    candidates.setdefault(job_id, set()).add(cluster)


def remote_path(value: PurePosixPath, executable: str) -> str:
    return str(value / executable)
