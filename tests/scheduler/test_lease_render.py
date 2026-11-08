"""Rendering a lease: its batch script, its steps' orders, and its sbatch request."""

from __future__ import annotations

import shlex
from datetime import timedelta
from pathlib import PurePosixPath
from typing import Any

import pytest
from scheduler_fixtures import ALLOCATION, allocation
from support.builders import resources, target

from servatus.campaign._config import Resources, Target, Task
from servatus.campaign._plan import slot_resources
from servatus.campaign._script import (
    _INTERRUPT,  # pyright: ignore[reportPrivateUsage]
    _START,  # pyright: ignore[reportPrivateUsage]
    _TRAP,  # pyright: ignore[reportPrivateUsage]
    Chained,
    order_directory,
    printf_literal,
    render_batch,
    render_lease,
    render_order,
    sbatch_argv,
)
from servatus.errors import ConfigurationError

LOG = "/cluster/logs/project"
LEASE = ALLOCATION
STEP = allocation(7)
SIGNALLED = resources(gpus=0, time_limit="02:00:00", signal_before_end="00:05:00")
DIRECT = target(container=None, gpu_gres=None, max_gpus_per_allocation=0)
STARTED = '/bin/sh -c \'command : > "$1" || exit 1; shift; exec "$@"\' servatus-start "$m.start"'
PRELUDE = (
    "#!/bin/sh\n"
    "set -u\n"
    "umask 077\n"
    "export PATH=/usr/bin:/bin\n"
    'job_id="${SLURM_JOB_ID:?Slurm did not set SLURM_JOB_ID}"\n'
)


def lease(target_value: Target | None = None, request: Resources = SIGNALLED) -> str:
    return render_lease(target_value or target(), request, LEASE).decode()


def chain(spent: int = 0, budget: timedelta | None = None) -> Chained:
    return Chained(1, budget, (1,), (timedelta(seconds=spent),))


def order(
    task: Task,
    target_value: Target = DIRECT,
    request: Resources = SIGNALLED,
    link: Chained | None = None,
) -> str:
    return render_order(target_value, request, task, STEP, link or chain()).decode()


def task(key: str = "t", *args: str, **options: Any) -> Task:
    return Task(key, args or ("/opt/app/run", "--flag"), **options)


def launch(script: str) -> str:
    """The one srun line of an order, without its trailing ``&``."""
    (line,) = [line for line in script.splitlines() if line.startswith("printf ")]
    assert line.endswith(" &")
    return line.removesuffix(" &")


# --- Lease batch script ----------------------------------------------------------------------


def test_a_lease_starts_with_the_campaign_prelude_and_its_marker() -> None:
    script = lease()
    plain = render_batch(DIRECT, SIGNALLED, (task(),), LEASE).decode()
    assert script.startswith(PRELUDE) and plain.startswith(PRELUDE)
    assert f'\nmarker={LOG}/{LEASE}."$job_id"\n' in script


def test_a_lease_needs_the_pre_limit_signal() -> None:
    with pytest.raises(ConfigurationError, match="a lease needs resources.signal_before_end"):
        lease(request=resources(gpus=0))


def test_a_lease_keeps_the_interrupt_trap_and_relays_the_limit_once_flagged() -> None:
    script = lease()
    assert "\n".join(_INTERRUPT) + "\n" in script
    assert (
        "limit() {\n"
        "  limited=1\n"
        "  trap '' USR1\n"
        '  command : > "$marker.limit"\n'
        "  kill -s USR1 0 2>/dev/null || :\n"
        "}\n"
        "trap limit USR1\n"
        f"{_START}\n"
    ) in script
    assert "\ngrace=300\n" in script and "\nlimited=0\n" in script
    assert script.count("sleep 5") == 2 and not script.endswith("exit 0\n")


def test_a_lease_quotes_its_log_root_and_order_directory() -> None:
    script = lease(target(log_root="/cluster/logs/my project"))
    assert f"\norders='/cluster/logs/my project/{LEASE}.d'\n" in script
    assert "\nlog='/cluster/logs/my project'\n" in script
    assert f"\nmarker='/cluster/logs/my project/{LEASE}'.\"$job_id\"\n" in script


def test_a_lease_script_is_bounded() -> None:
    with pytest.raises(ConfigurationError, match="exceeds target max_script_bytes"):
        lease(target(max_script_bytes=500))


def test_the_order_directory_belongs_to_the_lease() -> None:
    assert order_directory(PurePosixPath(LOG), LEASE) == PurePosixPath(f"{LOG}/{LEASE}.d")


@pytest.mark.parametrize(
    ("relay", "signal"), [(True, "--signal=B:USR1@300"), (False, "--signal=USR1@300")]
)
def test_relay_asks_for_the_signal_in_the_batch_shell(relay: bool, signal: str) -> None:
    assert sbatch_argv(DIRECT, SIGNALLED, 3, LEASE, relay=relay)[-1] == signal


# --- Campaign scripts stay as they were -------------------------------------------------------


def test_campaign_batch_scripts_are_unchanged_by_leases() -> None:
    assert (*_INTERRUPT, _START) == _TRAP
    plain = render_batch(DIRECT, resources(gpus=0), (task(),), ALLOCATION).decode()
    step = (
        f"printf '' | /opt/slurm/bin/srun --exclusive --exact --nodes=1 --ntasks=1 "
        f"--cpus-per-task=32 --mem=65536M --chdir=/cluster/work/project "
        f"--job-name=servatus-{ALLOCATION}-0 --export=ALL "
        f"--output={LOG}/{ALLOCATION}-%j-0.out --error={LOG}/{ALLOCATION}-%j-0.out "
        f"/usr/bin/env -i SERVATUS_TASK_KEY=t SERVATUS_ALLOCATION_ID={ALLOCATION} "
        'SERVATUS_SLOT=0 SERVATUS_JOB_ID="$job_id" /opt/app/run --flag'
    )
    assert plain == (
        f'{PRELUDE}marker={LOG}/{ALLOCATION}."$job_id"\n'
        "pids=\n"
        "interrupt() {\n"
        "  trap '' HUP INT TERM\n"
        '  command : > "$marker.stop"\n'
        "  kill -s TERM 0 2>/dev/null || :\n"
        '  for pid in $pids; do wait "$pid" 2>/dev/null || :; done\n'
        "  exit 1\n"
        "}\n"
        "trap interrupt HUP INT TERM\n"
        ': > "$marker.start" || exit 1\n'
        f'{{ {step}; code=$?; command : > "$marker.0.$code.end"; exit "$code"; }} &\n'
        'pids="$pids $!"\n'
        "status=0\n"
        'for pid in $pids; do wait "$pid" || status=1; done\n'
        'exit "$status"\n'
    )
    link = Chained(1, None, (2,), (timedelta(0),), allocation(8))
    chained = render_batch(DIRECT, SIGNALLED, (task(),), ALLOCATION, link).decode()
    assert (
        "limit() {\n"
        "  trap '' USR1\n"
        '  command : > "$marker.limit"\n'
        "  kill -s USR1 0 2>/dev/null || :\n"
        "}\n"
        "trap interrupt HUP INT TERM\n"
        "trap limit USR1\n"
        ': > "$marker.start" || exit 1\n'
        "started=$(date +%s)\n"
    ) in chained
    for script in (plain, chained):
        for word in ("  limited=1\n", "orders=", "servatus-start", "$m."):
            assert word not in script


# --- Orders ----------------------------------------------------------------------------------


def test_an_order_defines_its_run_time_and_one_backgrounded_launch() -> None:
    script = order(task(), link=chain(spent=90))
    lines = script.split("\n")
    assert lines[:2] == ["previous=90", "launch() {"] and lines[3:] == ["}", ""]
    assert lines[2] == f"{launch(script)} &" and script.count("srun") == 1


def test_an_order_names_its_step_and_log_after_the_step() -> None:
    step = launch(order(task()))
    assert f" --job-name=servatus-{STEP}-0 " in step
    assert f" --output={LOG}/{STEP}-%j-0.out --error={LOG}/{STEP}-%j-0.out " in step


@pytest.mark.parametrize(
    ("target_value", "gpus", "command"),
    [
        (DIRECT, 0, "/usr/bin/env -i "),
        (target(container=None), 1, "/bin/sh -c 'visible="),
        (target(), 0, "/usr/bin/apptainer exec --cleanenv "),
        (target(), 1, "/bin/sh -c 'APPTAINERENV_CUDA_VISIBLE_DEVICES="),
    ],
    ids=["direct", "direct-gpu", "apptainer", "apptainer-gpu"],
)
def test_an_order_runs_the_campaign_step_behind_its_own_start_marker(
    target_value: Target, gpus: int, command: str
) -> None:
    request = resources(gpus=gpus, time_limit="02:00:00", signal_before_end="00:05:00")
    link = chain(spent=30, budget=timedelta(hours=2))
    step = launch(order(task(), target_value, request, link))
    assert f" {STARTED} {command}" in step
    batch = render_batch(target_value, request, (task(),), STEP, link).decode()
    (line,) = [line for line in batch.splitlines() if line.startswith("{ printf ")]
    assert step.replace(f" {STARTED}", "", 1) == line.removeprefix("{ ").partition("; code=$?")[0]


@pytest.mark.parametrize(("target_value", "prefix"), [(DIRECT, ""), (target(), "APPTAINERENV_")])
def test_an_order_tells_its_step_it_is_link_1_of_1(target_value: Target, prefix: str) -> None:
    unbudgeted = launch(order(task("k"), target_value))
    for word in (
        f"{prefix}SERVATUS_TASK_KEY=k ",
        f"{prefix}SERVATUS_ALLOCATION_ID={STEP} ",
        f"{prefix}SERVATUS_SLOT=0 ",
        f"{prefix}SERVATUS_LINK=1 {prefix}SERVATUS_LINKS=1 ",
        f'{prefix}SERVATUS_JOB_ID="$job_id" {prefix}SERVATUS_RUN_TIME="$previous" ',
    ):
        assert word in unbudgeted
    assert "SERVATUS_BUDGET" not in unbudgeted
    budgeted = launch(order(task("k"), target_value, link=chain(budget=timedelta(hours=2))))
    assert f"{prefix}SERVATUS_BUDGET=7200 " in budgeted


def test_an_order_never_leaves_task_values_unquoted() -> None:
    args = ("/opt/app/run", "$(touch /tmp/x)", "a b", "it's", "`id`", ";", '"$HOME"')
    env = {"NOTE": "$HOME; rm -rf /", "LIST": "a,b c"}
    stdin = b"$(evil)\n%s\\'\x00"
    step = launch(order(task("key with space", *args, env=env, stdin=stdin)))
    head, _, words = step.partition(" | ")
    assert head == f"printf {printf_literal(stdin)}"
    tokens = shlex.split(words)
    assert tokens[-len(args) :] == list(args)
    assert {"NOTE=$HOME; rm -rf /", "LIST=a,b c", "SERVATUS_TASK_KEY=key with space"} <= set(tokens)
    for value in (*args[1:], *env.values()):
        assert shlex.quote(value) in step


def test_an_order_requests_the_slots_it_uses() -> None:
    request = resources(cpus=4, memory_mib=1000, gpus=1, signal_before_end="00:05:00")
    step = launch(order(task(), target(), slot_resources(request, 3)))
    assert " --cpus-per-task=12 --mem=3000M " in step and " --gres=gpu:a100:3 " in step


def test_an_order_is_bounded() -> None:
    with pytest.raises(ConfigurationError, match="exceeds target max_script_bytes"):
        order(task(), target(container=None, max_script_bytes=200))


def test_slot_resources_scale_the_per_task_request_but_not_its_time() -> None:
    request = resources(
        cpus=4, memory_mib=1000, gpus=1, time_limit="02:00:00", signal_before_end="00:05:00"
    )
    assert slot_resources(request, 1) is request
    scaled = slot_resources(request, 3)
    assert (scaled.cpus, scaled.memory_mib, scaled.gpus) == (12, 3000, 3)
    assert scaled.time_limit == timedelta(hours=2)
    assert scaled.signal_before_end == timedelta(minutes=5)
