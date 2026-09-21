from __future__ import annotations

import base64
import os
import re
import selectors
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from ._errors import ObservationError, ReconciliationError

if TYPE_CHECKING:
    from ._model import ResourceRequest, SlurmTarget, Task


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
    retained: bool = False


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


_MAX_QUERY_ARGC = 32
_MAX_QUERY_JOBS = 16
_MAX_QUERY_ARG_BYTES = 16 * 1024
_MAX_QUERY_FIELD_BYTES = 4096
_MAX_QUERY_LINES = 128
_MAX_QUERY_OUTPUT_BYTES = 1024 * 1024
_SSH_TIMEOUT_SECONDS = 30.0
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_SITE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")

_QUEUED_STATES = frozenset(
    {
        "PENDING",
        "CONFIGURING",
        "REQUEUED",
        "RESIZING",
        "REQUEUE_HOLD",
        "REQUEUE_FED",
        "RESV_DEL_HOLD",
        "SPECIAL_EXIT",
    }
)
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
        "TIMEOUT",
    }
)


def render_script(
    target: SlurmTarget,
    resources: ResourceRequest,
    tasks: tuple[Task, ...],
    allocation_id: str,
) -> bytes:
    lines = [
        "#!/bin/sh",
        "set -u",
        "umask 077",
        "status=0",
        "pids=",
        'scratch=$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/servatus.XXXXXXXXXX") || exit 1',
        '[ -n "$scratch" ] && [ -d "$scratch" ] || exit 1',
        'cleanup() { saved=$?; trap - 0; /bin/rm -rf -- "$scratch" || saved=1; exit "$saved"; }',
        "trap cleanup 0",
        "interrupt() { trap '' HUP INT TERM; "
        'pids="$pids ${!:-}"; '
        'for pid in $pids; do kill "$pid" 2>/dev/null || :; done; '
        'for pid in $pids; do wait "$pid" 2>/dev/null || :; done; exit 1; }',
        "trap interrupt HUP INT TERM",
        "export SLURM_EXPORT_ENV=ALL",
    ]
    for slot, task in enumerate(tasks):
        payload = base64.b64encode(task.stdin).decode("ascii")
        lines.extend(
            (
                f"if ! /usr/bin/base64 -d > \"$scratch/{slot}\" <<'SERVATUS_PAYLOAD'",
                payload,
                "SERVATUS_PAYLOAD",
                "then exit 1; fi",
            )
        )
    srun = target.slurm_bin / "srun"
    for slot, task in enumerate(tasks):
        process = slot + 1
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
            container.extend(("--nv", "--env", "CUDA_DEVICE_ORDER=PCI_BUS_ID"))
            wrapper = (
                ': "${CUDA_VISIBLE_DEVICES:?Slurm did not set step GPU visibility}"; '
                f"exec {shlex.join(container)} "
                '--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "$@"'
            )
            command = (
                *step,
                "/bin/sh",
                "-c",
                wrapper,
                "servatus-gpu",
                str(target.image),
                *task.args,
            )
        else:
            command = (*step, *container, str(target.image), *task.args)
        lines.append(f'{shlex.join(command)} < "$scratch/{slot}" &')
        lines.append(f"pid_{process}=$!")
        lines.append(f'pids="$pids $pid_{process}"')
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


def preflight(target: SlurmTarget, argv: tuple[str, ...]) -> None:
    """Reject deterministic local command failures before persisting submission intent."""
    _bounded_ssh_command(target, argv)


def _run_ssh(target: SlurmTarget, argv: tuple[str, ...], stdin: bytes) -> Result:
    return _run_bounded_ssh(target, argv, stdin=stdin)


def _run_bounded_ssh(
    target: SlurmTarget,
    argv: tuple[str, ...],
    *,
    stdin: bytes = b"",
    max_stdout_bytes: int = _MAX_QUERY_OUTPUT_BYTES,
) -> Result:
    command = _bounded_ssh_command(target, argv)
    with tempfile.TemporaryFile() as source:
        source.write(stdin)
        source.seek(0)
        process = subprocess.Popen(
            command,
            stdin=source,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_ssh_environment(),
        )
        selector: selectors.BaseSelector | None = None
        try:
            assert process.stdout is not None
            assert process.stderr is not None
            selector = selectors.DefaultSelector()
            stdout_descriptor = process.stdout.fileno()
            stderr_descriptor = process.stderr.fileno()
            buffers = {stdout_descriptor: bytearray(), stderr_descriptor: bytearray()}
            limits = {
                stdout_descriptor: max_stdout_bytes,
                stderr_descriptor: _MAX_QUERY_OUTPUT_BYTES,
            }
            for stream in (process.stdout, process.stderr):
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
                    chunk = os.read(descriptor, min(65536, limit + 1 - len(buffer)))
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    buffer.extend(chunk)
                    if len(buffer) > limit:
                        raise ObservationError("SSH output exceeds its byte bound")
            returncode = process.wait(timeout=max(0.0, deadline - time.monotonic()))
            return Result(
                returncode, bytes(buffers[stdout_descriptor]), bytes(buffers[stderr_descriptor])
            )
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            try:
                if selector is not None:
                    selector.close()
            finally:
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
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
        raise ObservationError("SSH command exceeds its argument bounds")
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
    groups: dict[str | None, list[_AttemptQuery]] = {}
    for attempt in attempts:
        groups.setdefault(attempt.cluster, []).append(attempt)
    observations: dict[str, SchedulerObservation] = {}
    for group in groups.values():
        batch: list[_AttemptQuery] = []
        for attempt in group:
            if len(batch) == _MAX_QUERY_JOBS or any(
                item.job_id == attempt.job_id for item in batch
            ):
                observations.update(_query_batch(target, tuple(batch)))
                batch = []
            batch.append(attempt)
        if batch:
            observations.update(_query_batch(target, tuple(batch)))
    return tuple(observations[attempt.allocation_id] for attempt in attempts)


def _query_batch(
    target: SlurmTarget,
    attempts: tuple[_AttemptQuery, ...],
) -> dict[str, SchedulerObservation]:
    by_job = {str(attempt.job_id): attempt for attempt in attempts}
    job_ids = ",".join(by_job)
    cluster = attempts[0].cluster
    route = "--local" if cluster is None else f"--clusters={cluster}"
    result = _run_observation(
        target,
        (
            str(target.slurm_bin / "squeue"),
            "--noheader",
            "--jobs",
            job_ids,
            route,
            "--format=%i|%j|%k|%V|%T|%r|%S|%e",
        ),
    )
    active: dict[str, _ActiveJob] = {}
    if not _missing_active_job(result):
        if result.returncode != 0 or result.stderr:
            raise ObservationError("scheduler observation was unavailable")
        for line in _output_lines(result.stdout):
            fields = _decoded_fields(line, 8)
            attempt = by_job.get(fields[0])
            if attempt is None:
                raise ObservationError("scheduler query returned unrelated evidence")
            if attempt.allocation_id in active:
                raise ObservationError("scheduler query returned conflicting evidence")
            active[attempt.allocation_id] = _parse_active_row(fields, attempt)
    result = _run_observation(
        target,
        (
            str(target.slurm_bin / "sacct"),
            "--noheader",
            "--parsable2",
            "--allocations",
            "--duplicates",
            "--jobs",
            job_ids,
            "--name",
            ",".join(attempt.identity for attempt in attempts),
            "--starttime",
            min(attempt.window_start for attempt in attempts),
            route,
            "--format=JobIDRaw%64,Cluster%256,JobName%256,Comment%256,Submit%32,"
            "State%256,ExitCode%32,Reason%4096,Start%32,End%32",
        ),
    )
    if result.returncode != 0 or result.stderr:
        raise ObservationError("scheduler observation was unavailable")
    accounting: dict[str, list[_AccountingJob]] = {item.allocation_id: [] for item in attempts}
    for line in _output_lines(result.stdout):
        fields = _decoded_fields(line, 10)
        attempt = by_job.get(fields[0])
        if attempt is None:
            raise ObservationError("scheduler query returned unrelated evidence")
        accounting[attempt.allocation_id].append(_parse_accounting_row(fields, attempt))
    return {
        attempt.allocation_id: _combine_observations(
            active.get(attempt.allocation_id),
            _accounting_history(tuple(accounting[attempt.allocation_id]), attempt),
        )
        for attempt in attempts
    }


def _parse_active_row(fields: tuple[str, ...], attempt: _AttemptQuery) -> _ActiveJob:
    job_id_raw, name, comment = fields[:3]
    submitted, state, reason, started, ended = _normalized_fields(*fields[3:])
    _validate_job_evidence(job_id_raw, state, attempt)
    if not _active_identity_matches(name, comment, attempt.identity):
        raise ObservationError("scheduler query returned unrelated evidence")
    submitted_at = _required_timestamp(submitted)
    if submitted_at < attempt.window_start:
        raise ObservationError("scheduler query returned unrelated evidence")
    return _ActiveJob(
        submitted_at, state, _optional_field(reason), _timestamp(started), _timestamp(ended)
    )


def _accounting_history(
    rows: tuple[_AccountingJob, ...], attempt: _AttemptQuery
) -> _AccountingHistory | None:
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
    return decoded


def _normalized_fields(*values: str) -> tuple[str, ...]:
    return tuple(value.strip(" ") for value in values)


def _parse_accounting_row(fields: tuple[str, ...], attempt: _AttemptQuery) -> _AccountingJob:
    (
        job_id,
        cluster_raw,
        name,
        comment,
    ) = fields[:4]
    (
        submitted,
        state,
        exit_code,
        reason,
        started,
        ended,
    ) = _normalized_fields(*fields[4:])
    _validate_job_evidence(job_id, state, attempt)
    if not _accounting_identity_matches(name, comment, attempt.identity):
        raise ObservationError("scheduler query returned unrelated evidence")
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


def _validate_job_evidence(
    job_id: str,
    state: str,
    attempt: _AttemptQuery,
) -> None:
    if (
        not job_id.isascii()
        or not job_id.isdecimal()
        or int(job_id) != attempt.job_id
        or _state_base(state) is None
    ):
        raise ObservationError("scheduler query returned unrelated evidence")


def _active_identity_matches(name: str, comment: str, identity: str) -> bool:
    return name == identity and comment == identity


def _accounting_identity_matches(name: str, comment: str, identity: str) -> bool:
    return name == identity and (comment == identity or _optional_field(comment) is None)


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
    if (
        history is not None
        and active is not None
        and (
            active.submitted_at < history.original_submitted_at
            or active.submitted_at < history.latest.submitted_at
        )
    ):
        raise ObservationError("scheduler sources returned conflicting evidence")
    retained = (
        active is not None
        and _normalize_state(active.state)
        not in {AllocationState.SUCCEEDED, AllocationState.FAILED, AllocationState.CANCELLED}
    ) or (
        accounting is not None
        and _normalize_state(accounting.state) in {AllocationState.QUEUED, AllocationState.RUNNING}
    )
    primary = active or accounting
    if history is None and not retained:
        primary = None
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
        same_base = _state_base(active.state) == _state_base(accounting.state)
        compatible_active_variant = active_state is accounting_state and active_state in {
            AllocationState.QUEUED,
            AllocationState.RUNNING,
        }
        accounting_is_older = (
            active_state is AllocationState.RUNNING and accounting_state is AllocationState.QUEUED
        )
        if same_base or compatible_active_variant or accounting_is_older:
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
        retained,
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
    if squeue.returncode != 0 or sacct.returncode != 0 or squeue.stderr or sacct.stderr:
        raise ReconciliationError("scheduler identity query was unavailable")

    candidates: dict[int, set[str | None]] = {}
    for job_id, name, comment in _identity_rows(squeue.stdout, 3):
        if not _active_identity_matches(name, comment, job_name):
            raise ReconciliationError("scheduler query returned unrelated allocation identity")
        _add_candidate(candidates, job_id, None)
    for job_id, name, comment, cluster_raw in _identity_rows(sacct.stdout, 4):
        if not _accounting_identity_matches(name, comment, job_name):
            raise ReconciliationError("scheduler query returned unrelated allocation identity")
        _add_candidate(candidates, job_id, cluster_raw or None)

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
    if (
        not raw_job_id.isascii()
        or not raw_job_id.isdecimal()
        or int(raw_job_id) <= 0
        or (cluster is not None and _SITE_TOKEN.fullmatch(cluster) is None)
    ):
        raise ReconciliationError("scheduler query returned unrelated allocation identity")
    job_id = int(raw_job_id)
    candidates.setdefault(job_id, set()).add(cluster)


def _identity_rows(output: bytes, expected: int) -> tuple[tuple[str, ...], ...]:
    try:
        return tuple(_decoded_fields(line, expected) for line in _output_lines(output))
    except ObservationError as error:
        raise ReconciliationError("scheduler identity query returned malformed evidence") from error
