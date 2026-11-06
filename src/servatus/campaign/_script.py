"""Pure rendering of one allocation: names, log paths, the ``sbatch`` argv, and the batch script,
plus ``check_command``, the deterministic bounds every scheduler command must meet.

The batch script needs only a POSIX shell and its builtins on the compute node: no scratch files,
no ``mktemp``, ``base64``, or ``rm``. Each Task becomes one exact ``srun`` step whose stdin is a
single-quoted ``printf`` literal, whose environment is rebuilt from nothing, and whose name binds
it to the allocation identity and slot. The script records its run in empty marker files in
``log_root`` (see ``_evidence``): a start marker before any step, one end marker per returned step,
and a stop marker when it is interrupted.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from datetime import timedelta
from pathlib import PurePosixPath

from ..errors import ConfigurationError
from ._codec import format_duration
from ._config import Apptainer, Resources, Target, Task

# Printable ASCII is copied verbatim except printf's own metacharacters and the quote itself.
_VERBATIM = frozenset(range(0x20, 0x7F)) - {ord("%"), ord("\\"), ord("'")}
_ENV = "/usr/bin/env"
_SHELL = "/bin/sh"
MAX_ARGS = 32
MAX_COMMAND_BYTES = 16 * 1024
MAX_FIELD_BYTES = 4096


def check_command(argv: Sequence[str]) -> tuple[str, ...]:
    """Check the deterministic bounds of one command without running it.

    At most 32 arguments (the first a nonempty program), 4 KiB per argument, and 16 KiB in total,
    all UTF-8 text without NUL.
    Raises ``ConfigurationError``, so callers can reject a command before recording any intent.
    """
    if isinstance(argv, (str, bytes)):
        raise ConfigurationError("command must be a sequence of arguments")
    fields = tuple(argv)
    if not fields or any(not isinstance(field, str) for field in fields):
        raise ConfigurationError("command must be a nonempty sequence of strings")
    if not fields[0]:
        raise ConfigurationError("command program must be nonempty")
    if len(fields) > MAX_ARGS:
        raise ConfigurationError(f"command exceeds its bound of {MAX_ARGS} arguments")
    total = 0
    for field in fields:
        if "\0" in field:
            raise ConfigurationError("command arguments cannot contain NUL")
        try:
            size = len(field.encode("utf-8"))
        except UnicodeEncodeError:
            raise ConfigurationError("command arguments must be valid UTF-8 text") from None
        if size > MAX_FIELD_BYTES:
            raise ConfigurationError(
                f"command argument exceeds its bound of {MAX_FIELD_BYTES} bytes"
            )
        total += size + 1
    if total > MAX_COMMAND_BYTES:
        raise ConfigurationError(f"command exceeds its bound of {MAX_COMMAND_BYTES} bytes")
    return fields


def job_name(allocation_id: str) -> str:
    """The immutable Slurm job name of one allocation."""
    return f"servatus-{allocation_id}"


def step_name(allocation_id: str, slot: int) -> str:
    """The ``srun --job-name`` of the step that runs the Task in ``slot``."""
    return f"{job_name(allocation_id)}-{slot}"


def log_path(
    log_root: PurePosixPath, allocation_id: str, job: int | str, slot: int | None = None
) -> PurePosixPath:
    """Combined stdout/stderr path: ``<root>/<id>-<job>.out`` or ``<root>/<id>-<job>-<slot>.out``.

    ``job`` is an accepted job number, or Slurm's ``%j`` placeholder inside rendered requests.
    """
    suffix = "" if slot is None else f"-{slot}"
    return log_root / f"{allocation_id}-{job}{suffix}.out"


def printf_literal(data: bytes) -> str:
    """One single-quoted ``printf`` format that writes exactly ``data`` (NUL-safe, builtin-only).

    Every escape uses three octal digits, so a following digit can never extend it.
    """
    return "'" + "".join(chr(byte) if byte in _VERBATIM else f"\\{byte:03o}" for byte in data) + "'"


def sbatch_argv(
    target: Target, resources: Resources, task_count: int, allocation_id: str
) -> tuple[str, ...]:
    """The complete ``sbatch`` request for one single-node allocation of ``task_count`` Tasks."""
    identity = job_name(allocation_id)
    output = log_path(target.log_root, allocation_id, "%j")
    argv = [
        str(target.slurm_bin / "sbatch"),
        "--parsable",
        "--no-requeue",
        "--export=NIL",
        "--nodes=1",
        f"--ntasks={task_count}",
        f"--cpus-per-task={resources.cpus}",
        f"--mem={task_count * resources.memory_mib}M",
        f"--time={format_duration(resources.time_limit)}",
        f"--partition={','.join(target.partitions)}",
        f"--chdir={target.work_root}",
        f"--job-name={identity}",
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
        argv.append(f"--gres={target.gpu_gres}:{task_count * resources.gpus}")
    if resources.signal_before_end is not None:
        seconds = resources.signal_before_end // timedelta(seconds=1)
        argv.append(f"--signal=USR1@{seconds}")
    return tuple(argv)


_PRELUDE = (
    "#!/bin/sh",
    "set -u",
    "umask 077",
    # --export=NIL leaves no PATH, and Apptainer's --nv finds the host NVIDIA binaries and
    # libraries (nvidia-smi, MPS, libcuda) on it; without one it binds nothing.
    "export PATH=/usr/bin:/bin",
    'job_id="${SLURM_JOB_ID:?Slurm did not set SLURM_JOB_ID}"',
)
# The trap signals the whole process group: each slot's subshell dies before it can write an end
# marker, so a step the trap interrupts is recorded as stopped, not as a failure of its own.
# `command :` keeps a failed marker write (quota, ESTALE) from exiting the shell and orphaning the
# steps.
_TRAP = (
    "pids=",
    "interrupt() {",
    "  trap '' HUP INT TERM",
    '  command : > "$marker.stop"',
    "  kill -s TERM 0 2>/dev/null || :",
    '  for pid in $pids; do wait "$pid" 2>/dev/null || :; done',
    "  exit 1",
    "}",
    "trap interrupt HUP INT TERM",
    ': > "$marker.start" || exit 1',
)
_EPILOGUE = (
    "status=0",
    'for pid in $pids; do wait "$pid" || status=1; done',
    'exit "$status"',
    "",
)
# Step-local GPU visibility exists only inside the step; read it there and fail when absent.
_GPU_VISIBILITY = '"${CUDA_VISIBLE_DEVICES:?Slurm did not set step GPU visibility}"'


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
    words.append(("SERVATUS_JOB_ID", '"$job_id"'))
    return tuple(words)


def _apptainer_command(
    container: Apptainer, target: Target, resources: Resources
) -> tuple[str, ...]:
    command = [str(container.executable), "exec", "--cleanenv"]
    for bind in (f"{target.work_root}:{target.work_root}", *container.binds):
        command.extend(("--bind", bind))
    command.extend(("--pwd", str(target.work_root)))
    if resources.gpus:
        command.append("--nv")
    command.append(str(container.image))
    return tuple(command)


def _launch(
    target: Target, resources: Resources, task: Task, allocation_id: str, slot: int
) -> tuple[str, str]:
    """The environment words before ``srun`` and the command ``srun`` runs for one Task.

    The direct launcher passes the environment as ``env -i NAME=VALUE...`` arguments. Apptainer
    receives it as ``APPTAINERENV_*`` assignments on ``srun`` itself, so the values never appear on
    a command line other users of the node can read. The batch shell expands only ``"$job_id"``;
    everything else is single-quoted. A GPU step runs through a shell that first reads the
    step-local visibility.
    """
    words = _environment(task, allocation_id, slot, resources.gpus)
    if target.container is None:
        program = task.args[0] if task.args else ""
        if not program.startswith("/") or "=" in program:
            raise ConfigurationError(
                f"Task {task.key!r}: without a container, args[0] must be an absolute program "
                "path without '='"
            )
        assignments = " ".join(f"{name}={word}" for name, word in words)
        if not resources.gpus:
            return "", f"{_ENV} -i {assignments} {shlex.join(task.args)}"
        wrapper = f'visible={_GPU_VISIBILITY}; exec {_ENV} -i CUDA_VISIBLE_DEVICES="$visible" "$@"'
        shell = shlex.join((_SHELL, "-c", wrapper, "servatus-gpu"))
        return "", f"{shell} {assignments} {shlex.join(task.args)}"
    if not task.args:
        raise ConfigurationError(
            f"Task {task.key!r}: an Apptainer Task needs args; args[0] is the program to run "
            "in the container (resolved on the container's PATH when relative)"
        )
    command = shlex.join((*_apptainer_command(target.container, target, resources), *task.args))
    prefix = " ".join(f"APPTAINERENV_{name}={word}" for name, word in words)
    if not resources.gpus:
        return prefix, command
    wrapper = f"APPTAINERENV_CUDA_VISIBLE_DEVICES={_GPU_VISIBILITY}; "
    wrapper += 'export APPTAINERENV_CUDA_VISIBLE_DEVICES; exec "$@"'
    return prefix, f"{shlex.join((_SHELL, '-c', wrapper, 'servatus-gpu'))} {command}"


def render_batch(
    target: Target, resources: Resources, tasks: Sequence[Task], allocation_id: str
) -> bytes:
    """The complete ``#!/bin/sh`` batch script that launches one exact step per Task.

    Steps start concurrently in slot order, each in a subshell that writes the slot's end marker
    as soon as its step returns; the script then waits for every step and exits nonzero if any
    failed. An interrupt writes the stop marker, then terminates every step.
    """
    srun = str(target.slurm_bin / "srun")
    marker = shlex.quote(str(target.log_root / allocation_id))
    lines: list[str] = [*_PRELUDE, f'marker={marker}."$job_id"', *_TRAP]
    for slot, task in enumerate(tasks):
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
            step.append(f"--gres={target.gpu_gres}:{resources.gpus}")
        prefix, command = _launch(target, resources, task, allocation_id, slot)
        words = f"{prefix} {shlex.join(step)} {command}".lstrip()
        end = f'code=$?; command : > "$marker.{slot}.$code.end"; exit "$code"'
        lines.append(f"{{ printf {printf_literal(task.stdin)} | {words}; {end}; }} &")
        lines.append('pids="$pids $!"')
    lines.extend(_EPILOGUE)
    script = "\n".join(lines).encode("utf-8")
    if len(script) > target.max_script_bytes:
        raise ConfigurationError(
            f"rendered batch script ({len(script)} bytes) exceeds target max_script_bytes "
            f"({target.max_script_bytes})"
        )
    return script
