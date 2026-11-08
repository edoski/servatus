"""Pure rendering of one allocation: names, log paths, the ``sbatch`` argv, and the batch script,
plus ``check_command``, the deterministic bounds every scheduler command must meet.

The batch script needs only a POSIX shell and its builtins on the compute node: no scratch files,
no ``mktemp``, ``base64``, or ``rm``. Each Task becomes one exact ``srun`` step whose stdin is a
single-quoted ``printf`` literal, whose environment is rebuilt from nothing, and whose name binds
it to the allocation identity and slot. The script records its run in empty marker files in
``log_root`` (see ``_evidence``): a start marker before any step, one end marker per returned step,
and a stop marker when it is interrupted.

A chain link (``Chained``) after the first launches a slot only once its handoff marker exists
and holds a plain decimal; anything else counts as no handoff.
A link that hands some slot on to the next link gets the pre-limit signal in its batch shell
(``--signal=B:USR1@N``): a trap writes a limit marker and relays the signal to the steps, and each
slot that ``continues`` writes the next link's handoff marker after its step returned. That link
exits 0 only when no slot handed off, which lets Slurm delete the rest of the chain. Every other
allocation keeps the plain shape, with ``--signal=USR1@N`` sent to the steps by Slurm.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import PurePosixPath

from ..errors import ConfigurationError
from ._codec import format_duration
from ._config import YIELD_EXIT, Apptainer, Resources, Target, Task

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


@dataclass(frozen=True, slots=True)
class Chained:
    """One chain link's script inputs, all decided at planning.

    ``index`` is the link's 1-based position. ``counts`` holds each slot's own link count, so a
    slot runs in this link only while ``index <= counts[slot]`` and hands off while ``index <
    counts[slot]``. ``spent`` is each slot's run time before the chain, the run time link 1
    reports to its steps; later links read it from their handoff markers. ``next_id`` names the
    next link, which a link that hands off writes markers for.
    """

    index: int
    budget: timedelta | None
    counts: tuple[int, ...]
    spent: tuple[timedelta, ...]
    next_id: str | None = None

    def __post_init__(self) -> None:
        if len(self.spent) != len(self.counts) or not 1 <= self.index <= max(self.counts):
            raise ConfigurationError("a chain link needs one count and run time per slot")
        if (self.next_id is not None) != self.hands_off:
            raise ConfigurationError("a chain link names its next link exactly when it hands off")

    @property
    def hands_off(self) -> bool:
        """Whether some slot continues in the next link."""
        return any(self.index < count for count in self.counts)


def continues(code: int | None, *, interrupted: bool, limited: bool) -> bool:
    """Whether a slot of a link that hands off continues in the next link: the rule the rendered
    shell implements, given the step's exit ``code`` (``None`` when interrupted), whether the
    batch script was interrupted, and whether the pre-limit signal had arrived.

    Interrupted, it continues only after the pre-limit signal (a time limit, not a cancel).
    Otherwise exit 75 continues, and so does any nonzero exit after the signal; exit 0 is done.
    """
    if interrupted:
        return limited
    return code == YIELD_EXIT or (limited and code != 0)


def dependency_options(jobs: Sequence[int]) -> tuple[str, ...]:
    """The ``sbatch`` options of a link that follows ``jobs`` (every earlier link of its chain):
    it may start only once each ended without exit 0, and Slurm deletes it once any exits 0."""
    if not jobs:
        return ()
    return (
        f"--dependency=afternotok:{':'.join(map(str, jobs))}",
        "--kill-on-invalid-dep=yes",
    )


def sbatch_argv(
    target: Target,
    resources: Resources,
    task_count: int,
    allocation_id: str,
    *,
    relay: bool = False,
) -> tuple[str, ...]:
    """The complete ``sbatch`` request for one single-node allocation of ``task_count`` Tasks
    (a lease: its slots).

    A ``relay`` (a chain link that hands off, or a lease) asks for the pre-limit signal in its
    batch shell (``B:``)."""
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
        argv.append(f"--signal={'B:' if relay else ''}USR1@{seconds}")
    return tuple(argv)


_PRELUDE = (
    "#!/bin/sh",
    "set -u",
    "umask 077",
    # --export=NIL leaves no PATH, and Apptainer's --nv finds the host NVIDIA binaries and
    # libraries (nvidia-smi, MPS, libcuda) on it; without one it binds nothing.
    "export PATH=/usr/bin:/bin",
    'job_id="${SLURM_JOB_ID:?Slurm did not set SLURM_JOB_ID}"',
    'end_time="${SLURM_JOB_END_TIME:?Slurm did not set SLURM_JOB_END_TIME}"',
)
# The trap signals the whole process group: each slot's subshell dies before it can write an end
# marker, so a step the trap interrupts is recorded as stopped, not as a failure of its own.
# `command :` keeps a failed marker write (quota, ESTALE) from exiting the shell and orphaning the
# steps.
_INTERRUPT = (
    "pids=",
    "interrupt() {",
    "  trap '' HUP INT TERM",
    '  command : > "$marker.stop"',
    "  kill -s TERM 0 2>/dev/null || :",
    '  for pid in $pids; do wait "$pid" 2>/dev/null || :; done',
    "  exit 1",
    "}",
    "trap interrupt HUP INT TERM",
)
_START = ': > "$marker.start" || exit 1'
_TRAP = (*_INTERRUPT, _START)
# The pre-limit signal: record it, then relay it to the process group, where `srun` forwards it
# to each step.
_LIMIT = (
    "  trap '' USR1",
    '  command : > "$marker.limit"',
    "  kill -s USR1 0 2>/dev/null || :",
    "}",
)
_EPILOGUE = (
    "status=0",
    'for pid in $pids; do wait "$pid" || status=1; done',
    'exit "$status"',
    "",
)
# A trap interrupts `wait` with a made-up status (128 + N, or 1 in ksh), so `reap` waits again
# until the status repeats or the shell no longer knows the child (127).
_RELAY = (
    "pids=",
    "continued=0",
    "reap() {",
    '  while [ -n "$pids" ]; do',
    "    set -- $pids",
    '    wait "$1"',
    "    code=$?",
    '    while wait "$1" 2>/dev/null; again=$?; [ "$again" -ne 127 ] && [ "$again" -ne "$code" ]',
    "    do code=$again; done",
    "    shift",
    "    pids=$*",
    '    [ "$code" -eq 0 ] || continued=1',
    "  done",
    "}",
    "interrupt() {",
    "  trap '' HUP INT TERM USR1",
    '  command : > "$marker.stop"',
    "  kill -s TERM 0 2>/dev/null || :",
    "  reap",
    '  exit "$continued"',
    "}",
    "limit() {",
    *_LIMIT,
    "trap interrupt HUP INT TERM",
    "trap limit USR1",
    _START,
    "started=$(date +%s)",
)
_REAP = ("reap", 'exit "$continued"', "")
# Each handoff name is looked up exactly once, by the one link that may run it, on a node that
# never looked it up before, after the previous link ended; so the lookup cannot hit a cached
# negative entry and needs no wait or poll.
_GATE = (
    "handed() {",
    "  previous=",
    '  read -r previous 2>/dev/null < "$own.$1.go" || return 1',
    "  case $previous in ''|*[!0-9]*|0?*|??????????*) return 1 ;; esac",
    "}",
)
_HANDOFF = 'printf \'%s\\n\' "$((previous + $(date +%s) - started))" > "$next.{slot}.go"'
_STARTED = 'command : > "$1" || exit 1; shift; exec "$@"'
# Step-local GPU visibility exists only inside the step; read it there and fail when absent.
_GPU_VISIBILITY = '"${CUDA_VISIBLE_DEVICES:?Slurm did not set step GPU visibility}"'


def _environment(
    task: Task, allocation_id: str, slot: int, gpus: int, chain: Chained | None
) -> tuple[tuple[str, str], ...]:
    """Ordered ``NAME=<shell word>`` pairs; Servatus names override Task values for GPU steps.
    A chain link adds the slot's link position, the budget, and its run time so far."""
    values = dict(task.env)
    if gpus:
        values.pop("CUDA_VISIBLE_DEVICES", None)
        values["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    values["SERVATUS_TASK_KEY"] = task.key
    values["SERVATUS_ALLOCATION_ID"] = allocation_id
    values["SERVATUS_SLOT"] = str(slot)
    if chain is not None:
        values["SERVATUS_LINK"] = str(chain.index)
        values["SERVATUS_LINKS"] = str(chain.counts[slot])
        if chain.budget is not None:
            values["SERVATUS_BUDGET"] = str(chain.budget // timedelta(seconds=1))
    words = [(name, shlex.quote(value)) for name, value in values.items()]
    words.append(("SERVATUS_JOB_ID", '"$job_id"'))
    words.append(("SERVATUS_JOB_END_TIME", '"$end_time"'))
    if chain is not None:
        words.append(("SERVATUS_RUN_TIME", '"$previous"'))
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
    target: Target,
    resources: Resources,
    task: Task,
    allocation_id: str,
    slot: int,
    chain: Chained | None,
) -> tuple[str, str]:
    """The environment words before ``srun`` and the command ``srun`` runs for one Task.

    The direct launcher passes the environment as ``env -i NAME=VALUE...`` arguments. Apptainer
    receives it as ``APPTAINERENV_*`` assignments on ``srun`` itself, so the values never appear on
    a command line other users of the node can read. The batch shell expands only ``"$job_id"``,
    ``"$end_time"``, and a chain link's ``"$previous"``; everything else is single-quoted. A GPU
    step runs through a shell that first reads the step-local visibility.
    """
    words = _environment(task, allocation_id, slot, resources.gpus, chain)
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


def _step(
    target: Target,
    resources: Resources,
    task: Task,
    allocation_id: str,
    slot: int,
    chain: Chained | None,
    started: str | None = None,
) -> str:
    """``printf STDIN | [ENV] srun ... COMMAND`` for one slot. With ``started``, the shell word
    of a start marker, the step creates that marker itself before the command runs, and fails
    without running it when it cannot."""
    srun = str(target.slurm_bin / "srun")
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
    prefix, command = _launch(target, resources, task, allocation_id, slot, chain)
    if started is not None:
        command = f"{shlex.join((_SHELL, '-c', _STARTED, 'servatus-start'))} {started} {command}"
    words = f"{prefix} {shlex.join(step)} {command}".lstrip()
    return f"printf {printf_literal(task.stdin)} | {words}"


def _relayed(step: str, slot: int, hands_off: bool) -> str:
    """One slot of a link that hands off: it waits for its step whatever signal arrives, records
    the end unless interrupted, then writes the handoff when the slot ``continues``."""
    traps = "interrupted=0; limited=0; trap 'interrupted=1' HUP INT TERM; trap 'limited=1' USR1"
    if not hands_off:
        end = f'[ "$interrupted" = 1 ] || command : > "$marker.{slot}.$code.end"; exit 0'
        return f"{{ {traps}; {step}; code=$?; {end}; }} &"
    decide = (
        'go=0; if [ "$interrupted" = 1 ]; then go=$limited; '
        f'else command : > "$marker.{slot}.$code.end"; '
        f'if [ "$code" = {YIELD_EXIT} ] || {{ [ "$limited" = 1 ] && [ "$code" != 0 ]; }}; '
        "then go=1; fi; fi; "
        f'if [ "$go" = 1 ]; then {_HANDOFF.format(slot=slot)}; fi; exit "$go"'
    )
    return f"{{ {traps}; {step}; code=$?; {decide}; }} &"


def render_batch(
    target: Target,
    resources: Resources,
    tasks: Sequence[Task],
    allocation_id: str,
    chain: Chained | None = None,
) -> bytes:
    """The complete ``#!/bin/sh`` batch script that launches one exact step per Task.

    Steps start concurrently in slot order, each in a subshell that writes the slot's end marker
    as soon as its step returns. A plain allocation, or a chain link that hands nothing off, then
    waits for every step and exits nonzero if any failed; an interrupt writes the stop marker,
    then terminates every step. A chain link that hands off exits 0 only when no slot continues.
    """
    marker = shlex.quote(str(target.log_root / allocation_id))
    lines: list[str] = [*_PRELUDE, f'marker={marker}."$job_id"']
    if chain is not None and len(chain.counts) != len(tasks):
        raise ConfigurationError("a chain link needs one count and run time per slot")
    gated = chain is not None and chain.index > 1
    relay = chain is not None and chain.hands_off
    if gated:
        lines.append(f"own={marker}")
    if chain is not None and chain.next_id is not None:
        lines.append(f"next={shlex.quote(str(target.log_root / chain.next_id))}")
    lines.extend(_RELAY if relay else _TRAP)
    if gated:
        lines.extend(_GATE)
    for slot, task in enumerate(tasks):
        if chain is not None and chain.index > chain.counts[slot]:
            continue
        step = _step(target, resources, task, allocation_id, slot, chain)
        if chain is not None and relay:
            launch = _relayed(step, slot, chain.index < chain.counts[slot])
        else:
            end = f'code=$?; command : > "$marker.{slot}.$code.end"; exit "$code"'
            launch = f"{{ {step}; {end}; }} &"
        if gated:
            lines.extend((f"if handed {slot}; then", launch, 'pids="$pids $!"', "fi"))
            continue
        if chain is not None:
            lines.append(f"previous={chain.spent[slot] // timedelta(seconds=1)}")
        lines.extend((launch, 'pids="$pids $!"'))
    lines.extend(_REAP if relay else _EPILOGUE)
    return _bounded("\n".join(lines).encode("utf-8"), target)


# --- Leases ----------------------------------------------------------------------------------

POLL_SECONDS = 5
"""How often a lease looks for new orders."""


def order_directory(log_root: PurePosixPath, lease_id: str) -> PurePosixPath:
    """``<log_root>/<lease>.d``, owner-only: the client writes each step's order there as
    ``<step>.run``, and the client and the lease contend for ``<step>.claim``."""
    return log_root / f"{lease_id}.d"


# A step starts only once the lease's own claim on its order won: a symbolic link created
# without replacing, which the client's withdrawal contends for under the same name. The
# `readlink` makes a retransmitted creation whose first reply was lost still count as won.
_TAKE = (
    "taken() {",
    '  ln -s taken "$orders/$1.claim" 2>/dev/null ||',
    '    [ "$(readlink "$orders/$1.claim" 2>/dev/null)" = taken ]',
    "}",
)
# One step: it sources its order (its run time and `launch`), waits for its `srun` whatever
# signal arrives, and records its end. A stop request gets the pre-limit treatment: SIGUSR1,
# which srun forwards as-is, then after `grace` SIGTERM, which srun turns into SIGKILL; a step
# still waiting for room never started, so its srun gets SIGTERM at once. The watcher ignores
# USR1 and the step's shell catches it, so the relayed pre-limit signal kills neither, while
# srun, started from a shell that catches it, keeps forwarding it.
_RUN = (
    "run() {",
    "  trap : USR1",
    '  m=$log/$1."$job_id"',
    '  . "$orders/$1.run"',
    "  launch",
    "  step=$!",
    "  (",
    "    trap '' USR1",
    f'    until [ -e "$m.stop" ]; do sleep {POLL_SECONDS}; done',
    '    if [ -e "$m.start" ]; then',
    '      kill -s USR1 "$step" 2>/dev/null || exit 0',
    '      sleep "$grace"',
    "    fi",
    '    kill -s TERM "$step" 2>/dev/null',
    "  ) &",
    "  watch=$!",
    '  wait "$step"',
    "  code=$?",
    '  while wait "$step" 2>/dev/null; again=$?; [ "$again" -ne 127 ] && [ "$again" -ne "$code" ]',
    "  do code=$again; done",
    '  kill "$watch" 2>/dev/null',
    '  command : > "$m.0.$code.end"',
    "}",
)
# The lease takes an order at most once, never after its pre-limit signal, and never one whose
# claim the client won. Its loop never waits for a step, and it never exits on its own, so the
# next link of a chain of leases always becomes eligible when this one ends. Its idle sleep
# ignores the relayed pre-limit signal, which would otherwise kill it and make some shells log
# the death.
_SERVE = (
    "seen=",
    "while :; do",
    '  for order in "$orders"/*.run; do',
    '    [ "$limited" = 0 ] || break',
    "    id=${order##*/}",
    "    id=${id%.run}",
    "    case $id in ????????????????????????) ;; *) continue ;; esac",
    "    case $id in *[!0-9a-f]*) continue ;; esac",
    '    case $seen in *" $id"*) continue ;; esac',
    '    if taken "$id"; then',
    '      seen="$seen $id"',
    '      run "$id" &',
    '      pids="$pids $!"',
    '    elif [ -h "$orders/$id.claim" ]; then',
    '      seen="$seen $id"',
    "    fi",
    "  done",
    f"  ( trap '' USR1; exec sleep {POLL_SECONDS} ) &",
    '  wait "$!" || :',
    "done",
    "",
)


def render_lease(target: Target, resources: Resources, lease_id: str) -> bytes:
    """The ``#!/bin/sh`` batch script of one lease: it records its start like any allocation,
    makes its owner-only order directory, then serves orders until Slurm ends it. Interrupted,
    it records its stop and terminates every step, as an allocation does; at the pre-limit
    signal it records the limit, relays the signal to its steps, and takes no more orders."""
    if resources.signal_before_end is None:
        raise ConfigurationError("a lease needs resources.signal_before_end")
    prefix = target.log_root / lease_id
    orders = shlex.quote(str(order_directory(target.log_root, lease_id)))
    lines = [
        *_PRELUDE,
        f'marker={shlex.quote(str(prefix))}."$job_id"',
        f"log={shlex.quote(str(target.log_root))}",
        f"orders={orders}",
        f"grace={resources.signal_before_end // timedelta(seconds=1)}",
        "limited=0",
        *_INTERRUPT,
        "limit() {",
        "  limited=1",
        *_LIMIT,
        "trap limit USR1",
        _START,
        'mkdir -p "$orders" 2>/dev/null',
        '[ -d "$orders" ] && [ ! -h "$orders" ] && [ -O "$orders" ] || exit 1',
        *_TAKE,
        *_RUN,
        *_SERVE,
    ]
    return _bounded("\n".join(lines).encode("utf-8"), target)


def render_order(
    target: Target, resources: Resources, task: Task, step_id: str, chain: Chained
) -> bytes:
    """One step's order, which its lease sources: the run time the step reports
    (``previous``) and ``launch``, the Campaign's exact ``srun`` step of ``task`` with
    ``resources`` (the slots it uses), its environment from link 1 of 1 ``chain``, and a start
    marker the step writes itself, so a step waiting for room is not yet running."""
    step = _step(target, resources, task, step_id, 0, chain, '"$m.start"')
    lines = (
        f"previous={chain.spent[0] // timedelta(seconds=1)}",
        "launch() {",
        f"{step} &",
        "}",
        "",
    )
    return _bounded("\n".join(lines).encode("utf-8"), target)


def _bounded(script: bytes, target: Target) -> bytes:
    if len(script) > target.max_script_bytes:
        raise ConfigurationError(
            f"rendered batch script ({len(script)} bytes) exceeds target max_script_bytes "
            f"({target.max_script_bytes})"
        )
    return script
