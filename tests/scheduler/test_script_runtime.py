"""Run rendered batch scripts under real local shells with stub ``srun`` and ``apptainer``.

The stubs emulate only what the script relies on: ``srun`` sets step-local GPU visibility and
executes the step command with the batch environment; ``apptainer exec --cleanenv`` executes the
command with exactly the ``APPTAINERENV_`` variables. Nothing contacts Slurm or Apptainer.

The interrupt trap signals its whole process group, so every script runs in its own session.
"""

from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import pytest
from scheduler_fixtures import ALLOCATION
from support.builders import target

from servatus.campaign._config import Apptainer, Resources, Target, Task
from servatus.campaign._script import render_batch

JOB = "4242"
SHELLS = [
    shell for shell in ("/bin/sh", "/bin/dash", "/bin/bash", "/bin/ksh") if Path(shell).exists()
]
# The stub interpreter itself injects these on macOS (PEP 538 locale coercion, CoreFoundation).
INTERPRETER_NOISE = ("__CF_USER_TEXT_ENCODING", "LC_CTYPE")

SRUN = """#!{python}
import json, os, sys
args = sys.argv[1:]
options = []
while args and args[0].startswith("--"):
    options.append(args.pop(0))
config = json.load(open({config!r}))
name = next(item.split("=", 1)[1] for item in options if item.startswith("--job-name="))
record = {{
    "options": options,
    "path": os.environ.get("PATH"),
    "start_marker": os.path.exists(config["start"]),
}}
with open(os.path.join(config["dir"], name + ".srun.json"), "w") as handle:
    json.dump(record, handle)
if any(item.startswith("--gres=") for item in options) and not config["no_gpu"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-step-" + name.rsplit("-", 1)[1]
else:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
os.execv(args[0], args)
"""

APPTAINER = """#!{python}
import json, os, sys
args = sys.argv[1:]
assert args.pop(0) == "exec"
flags = []
while args[0].startswith("--"):
    flag = args.pop(0)
    flags.append(flag)
    if flag in ("--bind", "--pwd"):
        flags.append(args.pop(0))
image = args.pop(0)
prefix = "APPTAINERENV_"
env = {{name[len(prefix):]: value for name, value in os.environ.items() if name.startswith(prefix)}}
record = os.path.join({directory!r}, "apptainer-" + env["SERVATUS_SLOT"] + ".json")
with open(record, "w") as handle:
    json.dump({{"flags": flags, "image": image}}, handle)
os.execve(args[0], args, env)
"""

# On SIGTERM a worker records whether the batch script's stop marker already existed.
WORKER = """#!{python}
import json, os, signal, sys, time
out, mode = sys.argv[1], sys.argv[2]
def stop(*_):
    with open(out + ".stopped", "w") as handle:
        handle.write("stop marker" if os.path.exists({stop!r}) else "no stop marker")
    os._exit(143)
signal.signal(signal.SIGTERM, stop)
open(out + ".started", "w").close()
if mode == "block":
    time.sleep(60)
if mode == "slow":
    time.sleep(0.3)
data = b"" if mode == "early" else sys.stdin.buffer.read()
with open(out, "wb") as handle:
    handle.write(data)
environment = {{k: v for k, v in os.environ.items() if k not in {noise!r}}}
with open(out + ".json", "w") as handle:
    json.dump({{"argv": sys.argv[1:], "env": environment}}, handle)
sys.exit(7 if mode == "fail" else 0)
"""


def wait_until(condition: Callable[[], bool], what: str, seconds: float = 20) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, what
        time.sleep(0.02)


class Runtime:
    def __init__(self, root: Path, *, container: bool, no_gpu: bool = False) -> None:
        self.root = root
        self.bin = root / "bin"
        self.out = root / "out"
        self.records = root / "records"
        self.logs = root / "run logs"
        for directory in (self.bin, self.out, self.records, self.logs, root / "work"):
            directory.mkdir()
        config = root / "srun.json"
        config.write_text(
            json.dumps(
                {"dir": str(self.records), "no_gpu": no_gpu, "start": str(self.marker("start"))}
            )
        )
        python = sys.executable
        self._stub("srun", SRUN.format(python=python, config=str(config)))
        self._stub("apptainer", APPTAINER.format(python=python, directory=str(self.records)))
        self.worker = self._stub(
            "worker",
            WORKER.format(python=python, noise=INTERPRETER_NOISE, stop=str(self.marker("stop"))),
        )
        apptainer = Apptainer(
            executable=str(self.bin / "apptainer"),
            image="/images/work.sif",
            binds=("/data:/data:ro",),
        )
        self.target: Target = target(
            slurm_bin=str(self.bin),
            work_root=str(root / "work"),
            log_root=str(self.logs),
            container=apptainer if container else None,
            max_script_bytes=16 * 1024 * 1024,
        )

    def _stub(self, name: str, text: str) -> Path:
        path = self.bin / name
        path.write_text(text)
        path.chmod(0o700)
        return path

    def marker(self, suffix: str) -> Path:
        return self.logs / f"{ALLOCATION}.{JOB}.{suffix}"

    def markers(self) -> set[str]:
        prefix = f"{ALLOCATION}.{JOB}."
        return {path.name.removeprefix(prefix) for path in self.logs.iterdir()}

    def task(self, key: str, mode: str = "read", *extra: str, **options: Any) -> Task:
        return Task(key, (str(self.worker), str(self.out / key), mode, *extra), **options)

    def script(self, tasks: tuple[Task, ...], gpus: int = 0) -> Path:
        resources = Resources(cpus=1, memory_mib=100, time_limit="00:10:00", gpus=gpus)
        path = self.root / "batch.sh"
        path.write_bytes(render_batch(self.target, resources, tasks, ALLOCATION))
        return path

    @staticmethod
    def environment(**env: str) -> dict[str, str]:
        # Slurm's --export=NIL: the batch starts with (almost) nothing but SLURM_* variables.
        values = {"SLURM_JOB_ID": JOB, **env}
        return {key: value for key, value in values.items() if value}

    def run(self, shell: str, script: Path, **env: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [shell, str(script)],
            env=self.environment(**env),
            cwd=self.root,
            capture_output=True,
            timeout=60,
            check=False,
            start_new_session=True,
        )

    @contextmanager
    def spawn(self, shell: str, script: Path) -> Generator[subprocess.Popen[bytes]]:
        child = subprocess.Popen(
            [shell, str(script)],
            env=self.environment(),
            cwd=self.root,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            yield child
        finally:
            # Reap anything the test left behind, steps included.
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=20)
            if child.stderr is not None:
                child.stderr.close()

    def record(self, key: str) -> dict[str, Any]:
        return json.loads((self.out / f"{key}.json").read_text())

    def started(self, key: str) -> bool:
        return (self.out / f"{key}.started").exists()

    def stopped(self, key: str) -> str | None:
        path = self.out / f"{key}.stopped"
        return path.read_text() if path.exists() else None


TRICKY_ENV = {
    "KMP_AFFINITY": "granularity=fine,compact,1,0",
    "XLA_FLAGS": "--xla_gpu_autotune_level=2 --xla_dump_to=/tmp/x",
    "WANDB_TAGS": "sweep,lr=3e-4",
    "QUOTED": "\"double\" 'single' `tick` $HOME ${HOME} \\ back",
    "MULTILINE": "line one\nline two\n",
    "EMPTY": "",
    "UNICODE": "λ – ü",
    "ARGS": 'a="b",c=d',
    "CUDA_VISIBLE_DEVICES": "manual",
}
TRICKY_ARGS = ("a b", "'quoted'", '"double"', "$(touch never)", "", "λ", "x\ny", "*", "%s", "\\")


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize("container", [False, True], ids=["direct", "apptainer"])
@pytest.mark.parametrize("gpus", [0, 1])
def test_steps_receive_exact_stdin_argv_and_environment(
    tmp_path: Path, shell: str, container: bool, gpus: int
) -> None:
    runtime = Runtime(tmp_path, container=container)
    tricky = bytes(range(256)) * 16 + b"\x001\xff7%s%d\\c'\"\n\\0101"
    large = random.Random(7).randbytes(200_000)
    tasks = (
        runtime.task("tricky", "read", *TRICKY_ARGS, stdin=tricky, env=TRICKY_ENV),
        runtime.task("empty", stdin=b""),
        runtime.task("large", stdin=large, env={"OMP_NUM_THREADS": "4"}),
    )
    result = runtime.run(shell, runtime.script(tasks, gpus))
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "never").exists()
    assert runtime.markers() == {"start", "0.0.end", "1.0.end", "2.0.end"}
    for slot, task in enumerate(tasks):
        assert (runtime.out / task.key).read_bytes() == task.stdin
        record = runtime.record(task.key)
        assert record["argv"] == list(task.args[1:])
        expected = dict(task.env)
        if gpus:
            expected["CUDA_VISIBLE_DEVICES"] = f"GPU-step-{slot}"
            expected["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        expected.update(
            SERVATUS_TASK_KEY=task.key,
            SERVATUS_ALLOCATION_ID=ALLOCATION,
            SERVATUS_SLOT=str(slot),
            SERVATUS_JOB_ID=JOB,
        )
        assert record["env"] == expected
        step = json.loads((runtime.records / f"servatus-{ALLOCATION}-{slot}.srun.json").read_text())
        assert step["start_marker"], "a step started before the start marker existed"
        options: list[str] = step["options"]
        assert options[:5] == [
            "--exclusive",
            "--exact",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=1",
        ]
        assert f"--job-name=servatus-{ALLOCATION}-{slot}" in options
        assert "--export=ALL" in options
        # --export=NIL leaves no PATH; Apptainer --nv locates the host NVIDIA files on it.
        assert step["path"] == "/usr/bin:/bin"
        assert ("--gres=gpu:a100:1" in options) == bool(gpus)
        if container:
            launch = json.loads((runtime.records / f"apptainer-{slot}.json").read_text())
            work = str(tmp_path / "work")
            flags = ["--cleanenv", "--bind", f"{work}:{work}", "--bind", "/data:/data:ro"]
            flags += ["--pwd", work] + (["--nv"] if gpus else [])
            assert launch == {"flags": flags, "image": "/images/work.sif"}


@pytest.mark.parametrize("container", [False, True], ids=["direct", "apptainer"])
def test_missing_job_id_fails_before_any_marker_or_step(tmp_path: Path, container: bool) -> None:
    runtime = Runtime(tmp_path, container=container)
    result = runtime.run("/bin/sh", runtime.script((runtime.task("t"),)), SLURM_JOB_ID="")
    assert result.returncode != 0
    assert b"Slurm did not set SLURM_JOB_ID" in result.stderr
    assert not runtime.started("t")
    assert list(runtime.records.iterdir()) == []
    assert list(runtime.logs.iterdir()) == []


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize("problem", ["missing", "unwritable"])
def test_log_root_that_cannot_take_the_start_marker_starts_no_step(
    tmp_path: Path, shell: str, problem: str
) -> None:
    runtime = Runtime(tmp_path, container=False)
    script = runtime.script((runtime.task("t"),))
    if problem == "missing":
        runtime.logs.rmdir()
    else:
        if os.geteuid() == 0:
            pytest.skip("root can write to a read-only directory")
        runtime.logs.chmod(0o500)
    try:
        result = runtime.run(shell, script)
    finally:
        if runtime.logs.exists():
            runtime.logs.chmod(0o700)
    assert result.returncode != 0
    assert not runtime.started("t")
    assert list(runtime.records.iterdir()) == []
    assert not runtime.logs.exists() or list(runtime.logs.iterdir()) == []


@pytest.mark.parametrize("container", [False, True], ids=["direct", "apptainer"])
def test_missing_step_gpu_visibility_fails_without_starting_the_worker(
    tmp_path: Path, container: bool
) -> None:
    runtime = Runtime(tmp_path, container=container, no_gpu=True)
    result = runtime.run("/bin/sh", runtime.script((runtime.task("t"),), gpus=1))
    assert result.returncode != 0
    assert b"Slurm did not set step GPU visibility" in result.stderr
    assert not runtime.started("t")


@pytest.mark.parametrize("shell", SHELLS)
def test_failure_waits_for_every_sibling_and_aggregates(tmp_path: Path, shell: str) -> None:
    runtime = Runtime(tmp_path, container=False)
    tasks = (runtime.task("fail", "fail", stdin=b"f"), runtime.task("slow", "slow", stdin=b"slow"))
    result = runtime.run(shell, runtime.script(tasks))
    assert result.returncode == 1
    assert (runtime.out / "slow").read_bytes() == b"slow"
    assert (runtime.out / "fail").read_bytes() == b"f"
    assert runtime.markers() == {"start", "0.7.end", "1.0.end"}


def test_worker_may_close_large_stdin_early(tmp_path: Path) -> None:
    runtime = Runtime(tmp_path, container=False)
    tasks = (runtime.task("early", "early", stdin=b"x" * 2_000_000),)
    result = runtime.run("/bin/sh", runtime.script(tasks))
    assert result.returncode == 0, result.stderr
    assert runtime.markers() == {"start", "0.0.end"}


@pytest.mark.parametrize("shell", SHELLS)
def test_a_slot_that_returns_early_is_recorded_while_its_sibling_runs(
    tmp_path: Path, shell: str
) -> None:
    runtime = Runtime(tmp_path, container=False)
    tasks = (runtime.task("quick", "fail", stdin=b"q"), runtime.task("block", "block"))
    with runtime.spawn(shell, runtime.script(tasks)) as child:
        wait_until(lambda: runtime.marker("0.7.end").exists(), "early slot was never recorded")
        wait_until(lambda: runtime.started("block"), "sibling never started")
        assert child.poll() is None
        assert runtime.stopped("block") is None
        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=20) == 1
        wait_until(lambda: runtime.stopped("block") is not None, "sibling was never stopped")
    assert runtime.markers() == {"start", "0.7.end", "stop"}


@pytest.mark.parametrize("shell", SHELLS)
def test_interrupt_records_the_stop_then_terminates_every_step(tmp_path: Path, shell: str) -> None:
    runtime = Runtime(tmp_path, container=False)
    keys = ("b0", "b1", "b2")
    script = runtime.script(tuple(runtime.task(key, "block", stdin=b"x") for key in keys))
    with runtime.spawn(shell, script) as child:
        wait_until(lambda: all(runtime.started(key) for key in keys), "steps never started")
        assert runtime.markers() == {"start"}
        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=20) == 1
        assert runtime.marker("stop").exists()
        for key in keys:
            wait_until(lambda key=key: runtime.stopped(key) is not None, f"{key} kept running")
            assert runtime.stopped(key) == "stop marker", f"{key} was signalled before the stop"
        time.sleep(0.3)
    assert runtime.markers() == {"start", "stop"}, "an interrupted step recorded an outcome"


@pytest.mark.parametrize("shell", SHELLS)
def test_interrupt_before_any_launch_exits_without_errors(tmp_path: Path, shell: str) -> None:
    # Regression: the old trap expanded ${!:-}, an error under `set -u` before any background job.
    runtime = Runtime(tmp_path, container=False)
    script = runtime.script((runtime.task("t"),))
    text = script.read_bytes().replace(
        b"trap interrupt HUP INT TERM\n", b"trap interrupt HUP INT TERM\nkill -TERM $$\n", 1
    )
    script.write_bytes(text)
    result = runtime.run(shell, script)
    assert result.returncode == 1
    assert result.stderr == b""
    assert not runtime.started("t")
    assert runtime.markers() == {"stop"}
