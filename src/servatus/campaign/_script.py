"""Pure rendering of one allocation: names, log paths, the ``sbatch`` argv, and the batch script.

The batch script needs only a POSIX shell and its builtins on the compute node: no scratch files,
no ``mktemp``, ``base64``, or ``rm``. Each Task becomes one exact ``srun`` step whose stdin is a
single-quoted ``printf`` literal, whose environment is rebuilt from nothing, and whose name binds
it to the allocation identity and slot.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Sequence
from datetime import timedelta
from pathlib import PurePosixPath

from ..errors import ConfigurationError
from ._codec import format_duration
from ._config import Apptainer, Resources, Target, Task

_ALLOCATION = re.compile(r"[0-9a-f]{24}\Z")
# Printable ASCII is copied verbatim except printf's own metacharacters and the quote itself.
_VERBATIM = frozenset(range(0x20, 0x7F)) - {ord("%"), ord("\\"), ord("'")}
_ENV = "/usr/bin/env"
_SHELL = "/bin/sh"


def _check_allocation(allocation_id: str) -> str:
    if not isinstance(allocation_id, str) or _ALLOCATION.fullmatch(allocation_id) is None:
        raise ConfigurationError("allocation_id must be 24 lowercase hexadecimal digits")
    return allocation_id


def job_name(allocation_id: str) -> str:
    """The immutable Slurm job name (and comment) of one allocation."""
    return f"servatus-{_check_allocation(allocation_id)}"


def step_name(allocation_id: str, slot: int) -> str:
    """The ``srun --job-name`` of the step that runs the Task in ``slot``."""
    if type(slot) is not int or slot < 0:
        raise ConfigurationError("slot must be a non-negative integer")
    return f"{job_name(allocation_id)}-{slot}"


def log_path(
    log_root: PurePosixPath, allocation_id: str, job: int | str, slot: int | None = None
) -> PurePosixPath:
    """Combined stdout/stderr path: ``<root>/<id>-<job>.out`` or ``<root>/<id>-<job>-<slot>.out``.

    ``job`` is an accepted job number, or Slurm's ``%j`` placeholder inside rendered requests.
    """
    if not (type(job) is int and job > 0) and job != "%j":
        raise ConfigurationError("job must be a positive job number or '%j'")
    if slot is not None and (type(slot) is not int or slot < 0):
        raise ConfigurationError("slot must be a non-negative integer")
    suffix = "" if slot is None else f"-{slot}"
    return log_root / f"{_check_allocation(allocation_id)}-{job}{suffix}.out"


def printf_literal(data: bytes) -> str:
    """One single-quoted ``printf`` format that writes exactly ``data`` (NUL-safe, builtin-only).

    Every escape uses three octal digits, so a following digit can never extend it.
    """
    return "'" + "".join(chr(byte) if byte in _VERBATIM else f"\\{byte:03o}" for byte in data) + "'"


def _gres(target: Target, gpus: int) -> str:
    if target.gpu_gres is None:
        raise ConfigurationError("GPU resources require a target gpu_gres")
    return f"--gres={target.gpu_gres}:{gpus}"


def sbatch_argv(
    target: Target, resources: Resources, task_count: int, allocation_id: str
) -> tuple[str, ...]:
    """The complete ``sbatch`` request for one single-node allocation of ``task_count`` Tasks."""
    if type(task_count) is not int or task_count < 1:
        raise ConfigurationError("task_count must be a positive integer")
    identity = job_name(allocation_id)
    output = log_path(target.log_root, allocation_id, "%j")
    argv = [
        str(target.slurm_bin / "sbatch"),
        "--parsable",
        "--export=NIL",
        "--nodes=1",
        f"--ntasks={task_count}",
        f"--cpus-per-task={resources.cpus}",
        f"--mem={task_count * resources.memory_mib}M",
        f"--time={format_duration(resources.time_limit)}",
        f"--partition={','.join(target.partitions)}",
        f"--chdir={target.work_root}",
        f"--job-name={identity}",
        f"--comment={identity}",
        f"--output={output}",
        f"--error={output}",
    ]
    if target.account is not None:
        argv.append(f"--account={target.account}")
    if target.qos is not None:
        argv.append(f"--qos={target.qos}")
    if target.constraint is not None:
        argv.append(f"--constraint={target.constraint}")
    if resources.gpus:
        argv.append(_gres(target, task_count * resources.gpus))
    if resources.signal_before_end is not None:
        seconds = resources.signal_before_end // timedelta(seconds=1)
        argv.append(f"--signal=USR1@{seconds}")
    return tuple(argv)


_PRELUDE = (
    "#!/bin/sh",
    "set -u",
    "umask 077",
    "pids=",
    "interrupt() {",
    "  trap '' HUP INT TERM",
    '  for pid in $pids; do kill "$pid" 2>/dev/null || :; done',
    '  for pid in $pids; do wait "$pid" 2>/dev/null || :; done',
    "  exit 1",
    "}",
    "trap interrupt HUP INT TERM",
    'job_id="${SLURM_JOB_ID:?Slurm did not set SLURM_JOB_ID}"',
    'restart_count="${SLURM_RESTART_COUNT:-0}"',
)
_EPILOGUE = (
    "status=0",
    'for pid in $pids; do wait "$pid" || status=1; done',
    'exit "$status"',
    "",
)
# Step-local GPU visibility exists only inside the step; read it there and fail when absent.
_GPU_VISIBILITY = '"${CUDA_VISIBLE_DEVICES:?Slurm did not set step GPU visibility}"'
_DYNAMIC = {"SERVATUS_JOB_ID": '"$job_id"', "SERVATUS_RESTART_COUNT": '"$restart_count"'}


def _environment(
    task: Task, allocation_id: str, slot: int, gpus: int
) -> tuple[tuple[str, str], ...]:
    """Ordered ``NAME=<shell word>`` pairs; Servatus names override Task values for GPU steps."""
    values = dict(task.env)
    if gpus:
        values.pop("CUDA_VISIBLE_DEVICES", None)
        values["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    values["SERVATUS_TASK_KEY"] = task.key
    values["SERVATUS_ALLOCATION_ID"] = allocation_id
    values["SERVATUS_SLOT"] = str(slot)
    words = [(name, shlex.quote(value)) for name, value in values.items()]
    words.extend(_DYNAMIC.items())
    return tuple(words)


def _apptainer_step(
    container: Apptainer,
    target: Target,
    resources: Resources,
    task: Task,
    allocation_id: str,
    slot: int,
) -> tuple[Sequence[str], str]:
    run = [str(container.executable), "run", "--cleanenv"]
    for bind in (f"{target.work_root}:{target.work_root}", *container.binds):
        run.extend(("--bind", bind))
    run.extend(("--pwd", str(target.work_root)))
    if resources.gpus:
        run.append("--nv")
    run.append(str(container.image))
    run.extend(task.args)
    assignments = " ".join(
        f"APPTAINERENV_{name}={word}"
        for name, word in _environment(task, allocation_id, slot, resources.gpus)
    )
    if not resources.gpus:
        return run, assignments
    wrapper = f"APPTAINERENV_CUDA_VISIBLE_DEVICES={_GPU_VISIBILITY}; "
    wrapper += 'export APPTAINERENV_CUDA_VISIBLE_DEVICES; exec "$@"'
    return (_SHELL, "-c", wrapper, "servatus-gpu", *run), assignments


def _direct_step(
    resources: Resources, task: Task, allocation_id: str, slot: int
) -> tuple[Sequence[str], str, Sequence[str]]:
    program = task.args[0] if task.args else ""
    if not program.startswith("/") or "=" in program:
        raise ConfigurationError(
            f"Task {task.key!r}: without a container, args[0] must be an absolute program path "
            "without '='"
        )
    assignments = " ".join(
        f"{name}={word}" for name, word in _environment(task, allocation_id, slot, resources.gpus)
    )
    if not resources.gpus:
        return (_ENV, "-i"), assignments, task.args
    wrapper = f"visible={_GPU_VISIBILITY}; "
    wrapper += f'exec {_ENV} -i CUDA_VISIBLE_DEVICES="$visible" "$@"'
    return (_SHELL, "-c", wrapper, "servatus-gpu"), assignments, task.args


def render_batch(
    target: Target, resources: Resources, tasks: Sequence[Task], allocation_id: str
) -> bytes:
    """The complete ``#!/bin/sh`` batch script that launches one exact step per Task.

    Steps start concurrently in slot order; the script then waits for every step and exits
    nonzero if any failed. An interrupt kills and reaps only the recorded step processes.
    """
    frozen = tuple(tasks)
    if not frozen or any(not isinstance(task, Task) for task in frozen):
        raise ConfigurationError("an allocation needs at least one Task")
    _check_allocation(allocation_id)
    srun = str(target.slurm_bin / "srun")
    lines: list[str] = list(_PRELUDE)
    for slot, task in enumerate(frozen):
        path = log_path(target.log_root, allocation_id, "%j", slot)
        step = [
            srun,
            "--exclusive",
            "--exact",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={resources.cpus}",
            f"--mem={resources.memory_mib}M",
            f"--chdir={target.work_root}",
            f"--job-name={step_name(allocation_id, slot)}",
            "--export=ALL",
            f"--output={path}",
            f"--error={path}",
        ]
        if resources.gpus:
            step.append(_gres(target, resources.gpus))
        container = target.container
        if container is None:
            # Environment words follow `env -i`; the batch shell expands only "$job_id" and
            # "$restart_count" there, everything else is single-quoted.
            launcher, assignments, args = _direct_step(resources, task, allocation_id, slot)
            words = f"{shlex.join((*step, *launcher))} {assignments} {shlex.join(args)}"
        else:
            command, assignments = _apptainer_step(
                container, target, resources, task, allocation_id, slot
            )
            words = f"{assignments} {shlex.join((*step, *command))}"
        lines.append(f"printf {printf_literal(task.stdin)} | {words} &")
        lines.append('pids="$pids $!"')
    lines.extend(_EPILOGUE)
    script = "\n".join(lines).encode("utf-8")
    if len(script) > target.max_script_bytes:
        raise ConfigurationError(
            f"rendered batch script ({len(script)} bytes) exceeds target max_script_bytes "
            f"({target.max_script_bytes})"
        )
    return script
