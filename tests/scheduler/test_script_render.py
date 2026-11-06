from __future__ import annotations

import re
from datetime import timedelta
from pathlib import PurePosixPath

import pytest
from hypothesis import given
from hypothesis import strategies as st
from scheduler_fixtures import ALLOCATION
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


# --- sbatch argv -----------------------------------------------------------------------------


def test_sbatch_argv_is_complete_and_exact() -> None:
    request = resources(signal_before_end="00:05:00")
    full = target(qos="high", constraint="a100")
    assert sbatch_argv(full, request, 3, ALLOCATION) == (
        "/opt/slurm/bin/sbatch",
        "--parsable",
        "--no-requeue",
        "--export=NIL",
        "--nodes=1",
        "--ntasks=3",
        "--cpus-per-task=32",
        "--mem=196608M",
        "--time=3-00:00:00",
        "--partition=gpu",
        "--chdir=/cluster/work/project",
        f"--job-name=servatus-{ALLOCATION}",
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
    ):
        assert assignment in script
    assert 'job_id="${SLURM_JOB_ID:?' in script
    assert "RESTART_COUNT" not in script


def test_markers_are_named_by_allocation_and_runtime_job() -> None:
    script = render((task(),), log_root="/cluster/logs/my project")
    assert f"marker='/cluster/logs/my project/{ALLOCATION}'.\"$job_id\"\n" in script


def test_start_marker_is_written_after_the_trap_and_before_any_step() -> None:
    lines = render((task("a"), task("b"))).splitlines()
    job_id = next(i for i, line in enumerate(lines) if line.startswith("job_id="))
    marker = next(i for i, line in enumerate(lines) if line.startswith("marker="))
    trap = lines.index("trap interrupt HUP INT TERM")
    start = lines.index(': > "$marker.start" || exit 1')
    launches = [i for i, line in enumerate(lines) if line.endswith(" &")]
    assert job_id < marker < trap < start < min(launches)


def test_each_slot_writes_its_own_end_marker_in_its_background_subshell() -> None:
    lines = render(tuple(task(f"t{slot}") for slot in range(3))).splitlines()
    launches = [line for line in lines if line.endswith(" &")]
    for slot, line in enumerate(launches):
        assert line.startswith("{ printf ")
        assert line.endswith(
            f'; code=$?; command : > "$marker.{slot}.$code.end"; exit "$code"; }} &'
        )
    assert sum(".end" in line for line in lines) == 3


def test_interrupt_trap_records_the_stop_then_signals_the_process_group() -> None:
    script = render((task(),))
    assert "trap interrupt HUP INT TERM" in script
    # The handler first ignores further signals, so repeated signals cannot re-enter it.
    assert (
        "interrupt() {\n"
        "  trap '' HUP INT TERM\n"
        '  command : > "$marker.stop"\n'
        "  kill -s TERM 0 2>/dev/null || :\n"
        '  for pid in $pids; do wait "$pid" 2>/dev/null || :; done\n'
        "  exit 1\n"
        "}\n"
    ) in script
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
    assert " python train.py; code=$?;" in script
    assert "/usr/bin/apptainer exec --cleanenv " in script and " run " not in script


def test_apptainer_tasks_need_a_program_regardless_of_the_runscript() -> None:
    with pytest.raises(ConfigurationError, match="'t': an Apptainer Task needs args"):
        render((Task("t"),))


def test_render_rejects_a_script_over_its_byte_bound() -> None:
    with pytest.raises(ConfigurationError, match="exceeds target max_script_bytes"):
        render((task(),), max_script_bytes=100)
