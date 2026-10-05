from __future__ import annotations

import re
from datetime import timedelta
from pathlib import PurePosixPath

import pytest
from hypothesis import given
from hypothesis import strategies as st
from support.builders import resources, target

from servatus.campaign._codec import format_duration, parse_duration
from servatus.campaign._config import Apptainer, Resources, Task, whole_minutes
from servatus.campaign._script import (
    job_name,
    log_path,
    printf_literal,
    render_batch,
    sbatch_argv,
    step_name,
)
from servatus.errors import ConfigurationError

ALLOCATION = "00112233445566778899aabb"
LOG = "/cluster/logs/project"


def render(tasks: tuple[Task, ...], **changes: object) -> str:
    target_changes = {key: value for key, value in changes.items() if key != "gpus"}
    gpus = changes.get("gpus", 1)
    return render_batch(target(**target_changes), resources(gpus=gpus), tasks, ALLOCATION).decode()


def task(key: str = "t", *args: str, **options: object) -> Task:
    return Task(key, args or ("/opt/app/run", "--flag"), **options)  # pyright: ignore[reportArgumentType]


# --- Names and paths -------------------------------------------------------------------------


def test_names_and_log_paths_bind_the_allocation_identity() -> None:
    assert job_name(ALLOCATION) == f"servatus-{ALLOCATION}"
    assert step_name(ALLOCATION, 3) == f"servatus-{ALLOCATION}-3"
    root = PurePosixPath(LOG)
    assert log_path(root, ALLOCATION, 42) == PurePosixPath(f"{LOG}/{ALLOCATION}-42.out")
    assert log_path(root, ALLOCATION, 42, 0) == PurePosixPath(f"{LOG}/{ALLOCATION}-42-0.out")
    assert log_path(root, ALLOCATION, "%j", 1) == PurePosixPath(f"{LOG}/{ALLOCATION}-%j-1.out")


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: job_name("abc"), "allocation_id"),
        (lambda: job_name(ALLOCATION.upper()), "allocation_id"),
        (lambda: step_name(ALLOCATION, -1), "slot"),
        (lambda: step_name(ALLOCATION, True), "slot"),  # a bool is not a slot number
        (lambda: log_path(PurePosixPath(LOG), ALLOCATION, 0), "job must be"),
        (lambda: log_path(PurePosixPath(LOG), ALLOCATION, "../x"), "job must be"),
        (lambda: log_path(PurePosixPath(LOG), ALLOCATION, 1, -2), "slot"),
    ],
)
def test_names_and_paths_reject_invalid_parts(call: object, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        call()  # pyright: ignore[reportCallIssue]


# --- sbatch argv -----------------------------------------------------------------------------


def test_sbatch_argv_is_complete_and_exact() -> None:
    request = resources(signal_before_end="00:05:00")
    full = target(qos="high", constraint="a100")
    assert sbatch_argv(full, request, 3, ALLOCATION) == (
        "/opt/slurm/bin/sbatch",
        "--parsable",
        "--export=NIL",
        "--nodes=1",
        "--ntasks=3",
        "--cpus-per-task=32",
        "--mem=196608M",
        "--time=3-00:00:00",
        "--partition=gpu",
        "--chdir=/cluster/work/project",
        f"--job-name=servatus-{ALLOCATION}",
        f"--comment=servatus-{ALLOCATION}",
        f"--output={LOG}/{ALLOCATION}-%j.out",
        f"--error={LOG}/{ALLOCATION}-%j.out",
        "--account=research",
        "--qos=high",
        "--constraint=a100",
        "--gres=gpu:a100:3",
        "--signal=USR1@300",
    )


def test_cpu_request_omits_gpu_account_and_signal_options() -> None:
    argv = sbatch_argv(
        target(account=None, gpu_gres=None, max_gpus_per_allocation=0, partitions=("a", "b")),
        resources(gpus=0),
        1,
        ALLOCATION,
    )
    assert not [item for item in argv if item.startswith(("--gres", "--account", "--signal"))]
    assert "--partition=a,b" in argv


def test_time_limit_is_rounded_up_to_whole_minutes() -> None:
    request = Resources(cpus=1, memory_mib=1, time_limit="00:01:30")
    assert "--time=00:02:00" in sbatch_argv(target(), request, 1, ALLOCATION)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: sbatch_argv(target(), resources(), 0, ALLOCATION), "task_count"),
        (lambda: sbatch_argv(target(), resources(), 1, "nope"), "allocation_id"),
        (
            lambda: sbatch_argv(
                target(gpu_gres=None, max_gpus_per_allocation=0), resources(), 1, ALLOCATION
            ),
            "gpu_gres",
        ),
    ],
)
def test_sbatch_argv_rejects_invalid_requests(call: object, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        call()  # pyright: ignore[reportCallIssue]


durations = st.builds(
    timedelta,
    days=st.integers(0, 400),
    hours=st.integers(0, 23),
    minutes=st.integers(0, 59),
    seconds=st.integers(0, 59),
).filter(lambda value: value > timedelta(0))


@given(durations)
def test_requested_time_is_canonical_never_shorter_and_minute_aligned(value: timedelta) -> None:
    rounded = whole_minutes(value)
    request = Resources(cpus=1, memory_mib=1, time_limit=value)
    (time,) = [item for item in sbatch_argv(target(), request, 1, ALLOCATION) if "--time=" in item]
    text = time.removeprefix("--time=")
    assert text == format_duration(rounded)
    assert parse_duration(text) == rounded
    assert value <= rounded < value + timedelta(minutes=1)
    days, _, clock = text.rpartition("-")
    assert (days == "") == (rounded < timedelta(days=1))
    assert int(clock[:2]) < 24 or not days


# --- printf payloads -------------------------------------------------------------------------


def decode_printf_literal(literal: str) -> bytes:
    """Interpret a single-quoted printf format the way POSIX printf does (escapes only)."""
    assert literal.startswith("'") and literal.endswith("'")
    body = literal[1:-1]
    assert "'" not in body and "%" not in body
    output = bytearray()
    index = 0
    while index < len(body):
        if body[index] == "\\":
            digits = body[index + 1 : index + 4]
            assert re.fullmatch(r"[0-7]{3}", digits), digits
            output.append(int(digits, 8))
            index += 4
        else:
            assert 0x20 <= ord(body[index]) < 0x7F
            output.append(ord(body[index]))
            index += 1
    return bytes(output)


@given(st.binary(max_size=4096))
def test_printf_literal_round_trips_every_byte_sequence(data: bytes) -> None:
    assert decode_printf_literal(printf_literal(data)) == data


def test_printf_literal_keeps_printable_ascii_verbatim() -> None:
    assert printf_literal(b"run --lr 3e-4") == "'run --lr 3e-4'"
    assert printf_literal(b"%s'\\\x001") == "'\\045s\\047\\134\\0001'"


# --- Batch script ----------------------------------------------------------------------------


def test_script_starts_every_step_before_waiting_and_aggregates() -> None:
    script = render(tuple(task(f"t{index}") for index in range(4)))
    lines = script.splitlines()
    launches = [index for index, line in enumerate(lines) if line.endswith(" &")]
    assert len(launches) == 4
    first_wait = next(index for index, line in enumerate(lines) if line.startswith("for pid"))
    assert max(launches) < first_wait
    assert lines[-1] == 'exit "$status"'
    assert script.count("--exclusive --exact --nodes=1 --ntasks=1") == 4


def test_script_names_steps_and_logs_by_slot() -> None:
    script = render((task("a"), task("b")))
    for slot in range(2):
        assert f"--job-name=servatus-{ALLOCATION}-{slot}" in script
        assert f"--output={LOG}/{ALLOCATION}-%j-{slot}.out" in script
        assert f"--error={LOG}/{ALLOCATION}-%j-{slot}.out" in script
    assert "%j-2.out" not in script and ".err" not in script


def test_script_uses_only_absolute_tools_and_shell_builtins() -> None:
    for script in (render((task(),)), render((task(),), container=None, gpus=0)):
        for forbidden in (r"\bmktemp\b", r"\bbase64\b", r"\brm\b", r"--env\b", r"\$\{!"):
            assert re.search(forbidden, script) is None, forbidden
        for forbidden in ("#SBATCH", "--overlap", "--mem=0", "SLURM_EXPORT_ENV", "scratch"):
            assert forbidden not in script


def test_cpu_steps_emit_no_gpu_flags_and_gpu_steps_do() -> None:
    cpu = render((task(),), gpus=0)
    assert "--gres" not in cpu and "--nv" not in cpu and "CUDA" not in cpu
    gpu = render((task(),), gpus=2)
    assert "--gres=gpu:a100:2" in gpu and " --nv " in gpu
    assert "APPTAINERENV_CUDA_DEVICE_ORDER=PCI_BUS_ID" in gpu


def test_apptainer_binds_work_root_and_configured_binds() -> None:
    container = Apptainer(
        executable="/usr/bin/apptainer", image="/images/x.sif", binds=("/data", "/a:/b:ro")
    )
    script = render((task(),), container=container)
    assert (
        "/usr/bin/apptainer exec --cleanenv --bind /cluster/work/project:/cluster/work/project "
        "--bind /data --bind /a:/b:ro --pwd /cluster/work/project --nv /images/x.sif"
    ) in script


def test_task_environment_is_passed_as_quoted_assignments_never_as_flags() -> None:
    # Regression: `--env NAME=a,b` split values on commas inside Apptainer.
    env = {"KMP_AFFINITY": "granularity=fine,compact,1,0", "NOTE": 'it\'s "x"'}
    script = render((task(env=env),))
    assert "APPTAINERENV_KMP_AFFINITY=granularity=fine,compact,1,0 " in script
    assert "APPTAINERENV_NOTE='it'\"'\"'s \"x\"'" in script
    direct = render((task(env=env),), container=None, gpus=0)
    assert "/usr/bin/env -i KMP_AFFINITY=granularity=fine,compact,1,0 NOTE=" in direct


def test_servatus_variables_are_injected_with_runtime_job_identity() -> None:
    script = render((task("key with space"),), gpus=0)
    for assignment in (
        "APPTAINERENV_SERVATUS_TASK_KEY='key with space'",
        f"APPTAINERENV_SERVATUS_ALLOCATION_ID={ALLOCATION}",
        "APPTAINERENV_SERVATUS_SLOT=0",
        'APPTAINERENV_SERVATUS_JOB_ID="$job_id"',
        'APPTAINERENV_SERVATUS_RESTART_COUNT="$restart_count"',
    ):
        assert assignment in script
    assert 'job_id="${SLURM_JOB_ID:?' in script
    assert 'restart_count="${SLURM_RESTART_COUNT:-0}"' in script


def test_interrupt_trap_kills_only_recorded_pids() -> None:
    script = render((task(),))
    assert "trap interrupt HUP INT TERM" in script
    # The handler first ignores further signals, so repeated signals cannot re-enter it.
    assert "interrupt() {\n  trap '' HUP INT TERM\n" in script
    assert 'for pid in $pids; do kill "$pid"' in script
    assert 'pids="$pids $!"' in script


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((), "absolute program path"),
        (("python", "train.py"), "absolute program path"),
        (("./run",), "absolute program path"),
        (("/opt/a=b/run",), "without '='"),
    ],
)
def test_direct_launcher_requires_an_absolute_program(args: tuple[str, ...], message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        render((Task("t", args),), container=None, gpus=0)


def test_apptainer_launcher_accepts_relative_commands_inside_the_image() -> None:
    script = render((Task("t", ("python", "train.py")),))
    assert " python train.py &" in script
    assert "/usr/bin/apptainer exec --cleanenv " in script and " run " not in script


def test_apptainer_tasks_need_a_program_regardless_of_the_runscript() -> None:
    with pytest.raises(ConfigurationError, match="'t': an Apptainer Task needs args"):
        render((Task("t"),))


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: render(()), "at least one Task"),
        (lambda: render((task(),), max_script_bytes=100), "exceeds target max_script_bytes"),
        (
            lambda: render((task(),), gpu_gres=None, max_gpus_per_allocation=0, gpus=1),
            "gpu_gres",
        ),
        (
            lambda: render_batch(target(), resources(), (task(),), "short"),
            "allocation_id",
        ),
    ],
)
def test_render_rejects_invalid_allocations(call: object, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        call()  # pyright: ignore[reportCallIssue]
