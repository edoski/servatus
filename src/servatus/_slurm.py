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


class AllocationState(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class SchedulerObservation:
    state: AllocationState
    raw_state: str | None
    accounting_state: str | None
    exit_code: str | None
    reason: str | None
    started_at: str | None
    ended_at: str | None


@dataclass(frozen=True, slots=True)
class _ActiveJob:
    submitted_at: str
    state: str
    reason: str | None
    started_at: str | None
    ended_at: str | None


@dataclass(frozen=True, slots=True)
class _AccountingJob:
    cluster: str | None
    submitted_at: str
    state: str
    exit_code: str | None
    reason: str | None
    started_at: str | None
    ended_at: str | None


@dataclass(frozen=True, slots=True)
class _AccountingHistory:
    original_submitted_at: str
    latest: _AccountingJob


@dataclass(frozen=True, slots=True)
class _AttemptQuery:
    allocation_id: str
    job_id: int
    cluster: str | None
    window_start: str
    window_end: str

    @property
    def identity(self) -> str:
        return f"servatus-{self.allocation_id}"


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
    completed = subprocess.run(
        ["ssh", "-T", "-o", "BatchMode=yes", target.host, _remote_command(argv)],
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
    command = _bounded_ssh_command(target, argv)
    process = subprocess.Popen(
        command,
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


def _bounded_ssh_command(target: SlurmTarget, argv: tuple[str, ...]) -> tuple[str, ...]:
    command = ("ssh", "-T", "-o", "BatchMode=yes", target.host, _remote_command(argv))
    fields = (*command[:-1], *argv)
    if (
        len(argv) > _MAX_QUERY_ARGC
        or len(command) > _MAX_QUERY_ARGC
        or any(len(value.encode("utf-8")) > _MAX_QUERY_FIELD_BYTES for value in fields)
        or sum(len(value.encode("utf-8")) + 1 for value in command) > _MAX_QUERY_ARG_BYTES
    ):
        raise ObservationError("observation command exceeds its argument bounds")
    return command


def _remote_command(argv: tuple[str, ...]) -> str:
    return "/usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C TZ=UTC " + shlex.join(argv)


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


def query_attempts(
    target: SlurmTarget,
    attempts: tuple[_AttemptQuery, ...],
) -> tuple[SchedulerObservation, ...]:
    observations: list[SchedulerObservation] = []
    for attempt in attempts:
        active = _query_active(target, attempt)
        history = _query_accounting(target, attempt)
        observations.append(_combine_observations(active, history))
    return tuple(observations)


def _query_active(
    target: SlurmTarget,
    attempt: _AttemptQuery,
) -> _ActiveJob | None:
    result = _run_observation(
        target,
        (
            str(target.slurm_bin / "squeue"),
            "--noheader",
            "--jobs",
            str(attempt.job_id),
            "--local" if attempt.cluster is None else f"--clusters={attempt.cluster}",
            "--format=%i|%j|%k|%V|%T|%r|%S|%e",
        ),
    )
    if _missing_active_job(result):
        return None
    if result.returncode != 0 or result.stderr:
        raise ObservationError("scheduler observation was unavailable")
    rows = _output_lines(result.stdout)
    if len(rows) > 1:
        raise ObservationError("scheduler query returned conflicting evidence")
    if not rows:
        return None
    fields = _decoded_fields(rows[0], 8)
    job_id_raw, name, comment, submitted, state, reason, started, ended = fields
    _validate_job_identity(job_id_raw, name, comment, state, attempt)
    submitted_at = _required_timestamp(submitted)
    return _ActiveJob(
        submitted_at,
        state,
        _optional_field(reason),
        _timestamp(started),
        _timestamp(ended),
    )


def _query_accounting(
    target: SlurmTarget,
    attempt: _AttemptQuery,
) -> _AccountingHistory | None:
    result = _run_observation(
        target,
        (
            str(target.slurm_bin / "sacct"),
            "--noheader",
            "--parsable2",
            "--allocations",
            "--duplicates",
            "--jobs",
            str(attempt.job_id),
            "--name",
            attempt.identity,
            "--starttime",
            attempt.window_start,
            "--local" if attempt.cluster is None else f"--clusters={attempt.cluster}",
            "--format=JobIDRaw%64,Cluster%256,JobName%256,Comment%256,Submit%32,"
            "State%256,ExitCode%32,Reason%4096,Start%32,End%32",
        ),
    )
    if result.returncode != 0 or result.stderr:
        raise ObservationError("scheduler observation was unavailable")
    rows = tuple(_parse_accounting_row(line, attempt) for line in _output_lines(result.stdout))
    if not rows:
        return None
    anchor_indexes = [
        index
        for index, row in enumerate(rows)
        if attempt.window_start <= row.submitted_at <= attempt.window_end
    ]
    if anchor_indexes != [0]:
        raise ObservationError("scheduler accounting history is ambiguous")
    anchor = rows[0]
    previous = ""
    for row in rows:
        if row.cluster != anchor.cluster or row.submitted_at < anchor.submitted_at:
            raise ObservationError("scheduler accounting history is ambiguous")
        if previous and row.submitted_at <= previous:
            raise ObservationError("scheduler accounting history is ambiguous")
        previous = row.submitted_at
    return _AccountingHistory(anchor.submitted_at, rows[-1])


def _run_observation(target: SlurmTarget, argv: tuple[str, ...]) -> Result:
    try:
        return _run_bounded_ssh(target, argv)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise ObservationError("scheduler observation was unavailable") from error


def _missing_active_job(result: Result) -> bool:
    return result == Result(
        1,
        b"",
        b"slurm_load_jobs error: Invalid job id specified\n",
    )


def _output_lines(output: bytes) -> tuple[bytes, ...]:
    if len(output) > _MAX_QUERY_OUTPUT_BYTES:
        raise ObservationError("scheduler query output exceeds its byte bound")
    if output and not output.endswith(b"\n"):
        raise ObservationError("scheduler query output is partial")
    lines = () if not output else output[:-1].split(b"\n")
    if len(lines) > _MAX_QUERY_LINES:
        raise ObservationError("scheduler query output exceeds its line bound")
    return tuple(lines)


def _decoded_fields(line: bytes, expected: int) -> tuple[str, ...]:
    fields = line.split(b"|")
    if len(fields) != expected or any(len(field) > _MAX_QUERY_FIELD_BYTES for field in fields):
        raise ObservationError("scheduler query row is malformed")
    try:
        decoded = tuple(field.decode("utf-8", "strict") for field in fields)
    except UnicodeDecodeError as error:
        raise ObservationError("scheduler query row is malformed") from error
    if any(_CONTROL.search(value) for value in decoded):
        raise ObservationError("scheduler query row is malformed")
    return tuple(value.strip(" ") for value in decoded)


def _parse_accounting_row(line: bytes, attempt: _AttemptQuery) -> _AccountingJob:
    (
        job_id,
        cluster_raw,
        name,
        comment,
        submitted,
        state,
        exit_code,
        reason,
        started,
        ended,
    ) = _decoded_fields(line, 10)
    _validate_job_identity(job_id, name, comment, state, attempt)
    cluster = _optional_field(cluster_raw)
    if (attempt.cluster is not None and cluster != attempt.cluster) or (
        cluster is not None and _SITE_TOKEN.fullmatch(cluster) is None
    ):
        raise ObservationError("scheduler query returned unrelated evidence")
    return _AccountingJob(
        cluster,
        _required_timestamp(submitted),
        state,
        _exit_code(exit_code),
        _optional_field(reason),
        _timestamp(started),
        _timestamp(ended),
    )


def _validate_job_identity(
    job_id: str,
    name: str,
    comment: str,
    state: str,
    attempt: _AttemptQuery,
) -> None:
    if (
        not job_id.isascii()
        or not job_id.isdecimal()
        or int(job_id) != attempt.job_id
        or name != attempt.identity
        or comment != attempt.identity
        or _state_base(state) is None
    ):
        raise ObservationError("scheduler query returned unrelated evidence")


def _required_timestamp(value: str) -> str:
    timestamp = _timestamp(value)
    if timestamp is None:
        raise ObservationError("scheduler query timestamp is malformed")
    return timestamp


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


def _combine_observations(
    active: _ActiveJob | None,
    history: _AccountingHistory | None,
) -> SchedulerObservation:
    accounting = None if history is None else history.latest
    if history is None:
        active = None
    elif active is not None and (
        active.submitted_at < history.original_submitted_at
        or active.submitted_at < history.latest.submitted_at
    ):
        raise ObservationError("scheduler sources returned conflicting evidence")
    primary = active or accounting
    if primary is None:
        return SchedulerObservation(
            AllocationState.UNKNOWN,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    active_state = None if active is None else _normalize_state(active.state)
    accounting_state = None if accounting is None else _normalize_state(accounting.state)
    if (
        active is not None
        and accounting is not None
        and active.submitted_at == accounting.submitted_at
    ):
        accounting_terminal = accounting_state in {
            AllocationState.SUCCEEDED,
            AllocationState.FAILED,
            AllocationState.CANCELLED,
        }
        if active_state is accounting_state or (
            active_state is AllocationState.RUNNING and accounting_state is AllocationState.QUEUED
        ):
            primary = active
        elif (
            active_state is AllocationState.QUEUED and accounting_state is AllocationState.RUNNING
        ) or (
            active_state in {AllocationState.QUEUED, AllocationState.RUNNING}
            and accounting_terminal
        ):
            primary = accounting
        else:
            raise ObservationError("scheduler sources returned conflicting evidence")
    return SchedulerObservation(
        _normalize_state(primary.state),
        primary.state,
        None if accounting is None else accounting.state,
        None if accounting is None else accounting.exit_code,
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
