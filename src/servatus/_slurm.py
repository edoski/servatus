from __future__ import annotations

import base64
import os
import re
import selectors
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from ._errors import ObservationError, ReconciliationError

if TYPE_CHECKING:
    from ._campaign import JobReceipt, ResourceRequest, SlurmTarget, Task


@dataclass(frozen=True, slots=True)
class Result:
    returncode: int
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True, slots=True)
class IdentityMatch:
    job_id: int
    cluster: str | None


class AllocationState(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class SchedulerObservation:
    job_id: int
    cluster: str | None
    state: AllocationState
    raw_state: str | None
    accounting_state: str | None
    exit_code: str | None
    reason: str | None
    started_at: str | None
    ended_at: str | None


@dataclass(frozen=True, slots=True)
class _RawJob:
    job_id: int
    cluster: str | None
    state: str
    exit_code: str | None
    reason: str | None
    started_at: str | None
    ended_at: str | None


_MAX_JOB_IDS_PER_QUERY = 64
_MAX_QUERY_ARGC = 16
_MAX_QUERY_ARG_BYTES = 16 * 1024
_MAX_QUERY_FIELD_BYTES = 4096
_MAX_QUERY_LINES = 128
_MAX_QUERY_OUTPUT_BYTES = 1024 * 1024
_SSH_TIMEOUT_SECONDS = 30.0
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_SITE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")

_QUEUED_STATES = frozenset({"PENDING", "CONFIGURING", "REQUEUED", "RESIZING"})
_RUNNING_STATES = frozenset(
    {"RUNNING", "COMPLETING", "SIGNALING", "STAGE_OUT", "SUSPENDED", "STOPPED"}
)
_SUCCEEDED_STATES = frozenset({"COMPLETED"})
_CANCELLED_STATES = frozenset({"CANCELLED", "PREEMPTED", "REVOKED"})
_FAILED_STATES = frozenset(
    {
        "BOOT_FAIL",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }
)


def render_script(
    target: SlurmTarget,
    resources: ResourceRequest,
    tasks: tuple[Task, ...],
    allocation_id: str,
) -> bytes:
    lines = ["#!/bin/sh", "set -u", "status=0"]
    srun = target.slurm_bin / "srun"
    for slot, task in enumerate(tasks):
        process = slot + 1
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
            f"--output={_log_path(target.log_root, allocation_id, '%j', slot)}",
            f"--error={_log_path(target.log_root, allocation_id, '%j', slot)}",
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
        lines.append(f"pid_{process}=$!")
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
        f"--output={_log_path(target.log_root, allocation_id, '%j')}",
        f"--error={_log_path(target.log_root, allocation_id, '%j')}",
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


def _run_bounded_ssh(
    target: SlurmTarget,
    argv: tuple[str, ...],
    *,
    max_stdout_bytes: int = _MAX_QUERY_OUTPUT_BYTES,
) -> Result:
    if len(argv) > _MAX_QUERY_ARGC or any(
        len(value.encode("utf-8")) > _MAX_QUERY_FIELD_BYTES for value in argv
    ):
        raise ObservationError("scheduler query command exceeds its argument bounds")
    remote = "/usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C " + shlex.join(argv)
    if len(remote.encode("utf-8")) > _MAX_QUERY_ARG_BYTES:
        raise ObservationError("scheduler query command exceeds its byte bound")
    process = subprocess.Popen(
        ["ssh", "-T", "-o", "BatchMode=yes", target.host, remote],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_ssh_environment(),
    )
    assert process.stdout is not None
    assert process.stderr is not None
    streams = (process.stdout, process.stderr)
    selector = selectors.DefaultSelector()
    stdout_descriptor = process.stdout.fileno()
    stderr_descriptor = process.stderr.fileno()
    buffers = {stdout_descriptor: bytearray(), stderr_descriptor: bytearray()}
    limits = {
        stdout_descriptor: max_stdout_bytes,
        stderr_descriptor: _MAX_QUERY_OUTPUT_BYTES,
    }
    try:
        for stream in streams:
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ)
        deadline = time.monotonic() + _SSH_TIMEOUT_SECONDS
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired("ssh", _SSH_TIMEOUT_SECONDS)
            events = selector.select(remaining)
            if not events:
                raise subprocess.TimeoutExpired("ssh", _SSH_TIMEOUT_SECONDS)
            for key, _ in events:
                descriptor = key.fd
                buffer = buffers[descriptor]
                limit = limits[descriptor]
                chunk = os.read(
                    descriptor,
                    min(65536, limit + 1 - len(buffer)),
                )
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                buffer.extend(chunk)
                if len(buffer) > limit:
                    raise ObservationError("scheduler query output exceeds its byte bound")
        returncode = process.wait(timeout=max(0.0, deadline - time.monotonic()))
        return Result(
            returncode,
            bytes(buffers[stdout_descriptor]),
            bytes(buffers[stderr_descriptor]),
        )
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        selector.close()
        for stream in streams:
            stream.close()


def _log_path(
    log_root: PurePosixPath,
    allocation_id: str,
    job_id: int | str,
    slot: int | None = None,
) -> PurePosixPath:
    slot_suffix = "" if slot is None else f"-{slot}"
    return log_root / f"{allocation_id}-{job_id}{slot_suffix}.out"


def read_log_suffix(
    target: SlurmTarget,
    path: PurePosixPath,
    max_bytes: int,
) -> tuple[bytes, bool]:
    read_limit = max_bytes + 1
    failed = False
    result: Result | None = None
    try:
        result = _run_bounded_ssh(
            target,
            ("/usr/bin/tail", "-c", str(read_limit), "--", str(path)),
            max_stdout_bytes=read_limit,
        )
    except Exception:
        failed = True
    if failed or result is None or result.returncode != 0 or result.stderr:
        raise ObservationError("campaign log is unavailable")
    if len(result.stdout) > read_limit:
        raise ObservationError("campaign log is unavailable")
    if len(result.stdout) == read_limit:
        return result.stdout[-max_bytes:], True
    return result.stdout, False


def query_receipts(
    target: SlurmTarget,
    receipts: tuple[JobReceipt, ...],
) -> tuple[SchedulerObservation, ...]:
    unique = {(receipt.job_id, receipt.cluster) for receipt in receipts}
    observations: dict[tuple[int, str | None], SchedulerObservation] = {}
    groups: dict[str | None, list[int]] = {}
    for job_id, cluster in sorted(unique, key=lambda item: (item[1] or "", item[0])):
        groups.setdefault(cluster, []).append(job_id)
    for cluster, job_ids in groups.items():
        for offset in range(0, len(job_ids), _MAX_JOB_IDS_PER_QUERY):
            chunk = tuple(job_ids[offset : offset + _MAX_JOB_IDS_PER_QUERY])
            active = _query_rows(target, chunk, cluster, accounting=False)
            accounting = _query_rows(target, chunk, cluster, accounting=True)
            for job_id in chunk:
                key = (job_id, cluster)
                observations[key] = _combine_rows(
                    job_id,
                    cluster,
                    active.get(job_id),
                    accounting.get(job_id),
                )
    return tuple(observations[(receipt.job_id, receipt.cluster)] for receipt in receipts)


def _query_rows(
    target: SlurmTarget,
    job_ids: tuple[int, ...],
    cluster: str | None,
    *,
    accounting: bool,
) -> dict[int, _RawJob]:
    command = target.slurm_bin / ("sacct" if accounting else "squeue")
    argv = [str(command), "--noheader"]
    if accounting:
        argv.extend(("--parsable2", "--allocations"))
    argv.extend(("--jobs", ",".join(map(str, job_ids))))
    argv.append("--local" if cluster is None else f"--clusters={cluster}")
    if accounting:
        argv.append(
            "--format=JobIDRaw%64,Cluster%256,State%256,ExitCode%32,Reason%4096,Start%32,End%32"
        )
    else:
        argv.append("--format=%i|%T|%r|%S|%e")
    try:
        result = _run_bounded_ssh(target, tuple(argv))
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise ObservationError("scheduler observation was unavailable") from error
    if result.returncode != 0 or result.stderr:
        raise ObservationError("scheduler observation was unavailable")
    return _parse_query_output(
        result.stdout,
        frozenset(job_ids),
        cluster,
        accounting=accounting,
    )


def _parse_query_output(
    output: bytes,
    expected_job_ids: frozenset[int],
    expected_cluster: str | None,
    *,
    accounting: bool,
) -> dict[int, _RawJob]:
    if len(output) > _MAX_QUERY_OUTPUT_BYTES:
        raise ObservationError("scheduler query output exceeds its byte bound")
    if output and not output.endswith(b"\n"):
        raise ObservationError("scheduler query output is partial")
    lines = output.splitlines()
    if len(lines) > _MAX_QUERY_LINES:
        raise ObservationError("scheduler query output exceeds its line bound")
    rows: dict[int, _RawJob] = {}
    for line in lines:
        fields = line.split(b"|")
        expected_fields = 7 if accounting else 5
        if len(fields) != expected_fields or any(
            len(field) > _MAX_QUERY_FIELD_BYTES for field in fields
        ):
            raise ObservationError("scheduler query row is malformed")
        try:
            decoded = tuple(field.decode("utf-8", "strict").strip() for field in fields)
        except UnicodeDecodeError as error:
            raise ObservationError("scheduler query row is malformed") from error
        if len(decoded) == 7:
            job_id_raw, cluster_raw, state_raw, exit_raw, reason_raw, start_raw, end_raw = decoded
            cluster = _optional_field(cluster_raw)
            exit_code = _exit_code(exit_raw)
        else:
            job_id_raw, state_raw, reason_raw, start_raw, end_raw = decoded
            cluster = expected_cluster
            exit_code = None
        if not job_id_raw.isascii() or not job_id_raw.isdecimal():
            raise ObservationError("scheduler query row has an invalid job identity")
        job_id = int(job_id_raw)
        if (
            job_id not in expected_job_ids
            or (expected_cluster is not None and cluster != expected_cluster)
            or (cluster is not None and _SITE_TOKEN.fullmatch(cluster) is None)
            or _state_base(state_raw) is None
            or any(_CONTROL.search(value) for value in decoded)
        ):
            raise ObservationError("scheduler query returned unrelated evidence")
        row = _RawJob(
            job_id,
            cluster,
            state_raw,
            exit_code,
            _optional_field(reason_raw),
            _timestamp(start_raw),
            _timestamp(end_raw),
        )
        existing = rows.get(job_id)
        if existing is not None and existing != row:
            raise ObservationError("scheduler query returned conflicting evidence")
        rows[job_id] = row
    return rows


def _optional_field(value: str) -> str | None:
    return None if value in {"", "None", "Unknown", "N/A"} else value


def _exit_code(value: str) -> str | None:
    if value in {"", "None", "Unknown", "N/A"}:
        return None
    parts = value.split(":")
    if len(parts) != 2 or any(not part.isascii() or not part.isdecimal() for part in parts):
        raise ObservationError("scheduler query exit code is malformed")
    return value


def _timestamp(value: str) -> str | None:
    if value in {"", "None", "Unknown", "N/A"}:
        return None
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
    except ValueError as error:
        raise ObservationError("scheduler query timestamp is malformed") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%S") != value:
        raise ObservationError("scheduler query timestamp is malformed")
    return value


def _normalize_state(raw: str) -> AllocationState:
    base = _state_base(raw)
    if base in _QUEUED_STATES:
        return AllocationState.QUEUED
    if base in _RUNNING_STATES:
        return AllocationState.RUNNING
    if base in _SUCCEEDED_STATES:
        return AllocationState.SUCCEEDED
    if base in _CANCELLED_STATES:
        return AllocationState.CANCELLED
    if base in _FAILED_STATES:
        return AllocationState.FAILED
    return AllocationState.UNKNOWN


def _state_base(raw: str) -> str | None:
    base = raw.split(" ", 1)[0].removesuffix("+")
    return base if _SITE_TOKEN.fullmatch(base) is not None else None


def _combine_rows(
    job_id: int,
    cluster: str | None,
    active: _RawJob | None,
    accounting: _RawJob | None,
) -> SchedulerObservation:
    primary = active or accounting
    if primary is None:
        return SchedulerObservation(
            job_id,
            cluster,
            AllocationState.UNKNOWN,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    if (
        active is not None
        and accounting is not None
        and active.cluster is not None
        and accounting.cluster is not None
        and active.cluster != accounting.cluster
    ):
        raise ObservationError("scheduler sources returned conflicting identities")
    active_state = None if active is None else _normalize_state(active.state)
    if (
        active is not None
        and accounting is not None
        and active_state not in {AllocationState.QUEUED, AllocationState.RUNNING}
        and _state_base(active.state) != _state_base(accounting.state)
    ):
        raise ObservationError("scheduler sources returned conflicting evidence")
    return SchedulerObservation(
        job_id,
        cluster,
        _normalize_state(primary.state),
        primary.state,
        None if accounting is None else accounting.state,
        primary.exit_code or (None if accounting is None else accounting.exit_code),
        primary.reason or (None if accounting is None else accounting.reason),
        primary.started_at or (None if accounting is None else accounting.started_at),
        primary.ended_at or (None if accounting is None else accounting.ended_at),
    )


_RECEIPT = re.compile(rb"([1-9][0-9]*)(?:;([A-Za-z0-9][A-Za-z0-9._-]*))?\n?\Z")


def parse_receipt(output: bytes) -> tuple[int, str | None]:
    match = _RECEIPT.fullmatch(output)
    if match is None:
        raise ValueError("sbatch did not return one positive job identity")
    job_id = int(match.group(1))
    cluster_bytes = match.group(2)
    return job_id, None if cluster_bytes is None else cluster_bytes.decode("ascii")


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
