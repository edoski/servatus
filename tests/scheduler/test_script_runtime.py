"""Run rendered batch scripts under real local shells with stub ``srun`` and ``apptainer``.

The stubs emulate only what the script relies on: ``srun`` sets step-local GPU visibility and
executes the step command with the batch environment; ``apptainer run --cleanenv`` executes the
command with exactly the ``APPTAINERENV_`` variables. Nothing contacts Slurm or Apptainer.
"""

from __future__ import annotations

import json
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from support.builders import target

from servatus.campaign._config import Apptainer, Resources, Target, Task
from servatus.campaign._script import render_batch

ALLOCATION = "00112233445566778899aabb"
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
with open(os.path.join(config["dir"], name + ".srun.json"), "w") as handle:
    json.dump({{"options": options}}, handle)
if any(item.startswith("--gres=") for item in options) and not config["no_gpu"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-step-" + name.rsplit("-", 1)[1]
else:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
os.execv(args[0], args)
"""

APPTAINER = """#!{python}
import json, os, sys
args = sys.argv[1:]
assert args.pop(0) == "run"
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

WORKER = """#!{python}
import json, os, signal, sys, time
out, mode = sys.argv[1], sys.argv[2]
def stop(*_):
    open(out + ".stopped", "w").close()
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


class Runtime:
    def __init__(self, root: Path, *, container: bool, no_gpu: bool = False) -> None:
        self.root = root
        self.bin = root / "bin"
        self.out = root / "out"
        self.records = root / "records"
        for directory in (self.bin, self.out, self.records, root / "work"):
            directory.mkdir()
        config = root / "srun.json"
        config.write_text(json.dumps({"dir": str(self.records), "no_gpu": no_gpu}))
        python = sys.executable
        self._stub("srun", SRUN.format(python=python, config=str(config)))
        self._stub("apptainer", APPTAINER.format(python=python, directory=str(self.records)))
        self.worker = self._stub("worker", WORKER.format(python=python, noise=INTERPRETER_NOISE))
        apptainer = Apptainer(
            executable=str(self.bin / "apptainer"),
            image="/images/work.sif",
            binds=("/data:/data:ro",),
        )
        self.target: Target = target(
            slurm_bin=str(self.bin),
            work_root=str(root / "work"),
            log_root=str(root / "logs"),
            container=apptainer if container else None,
            max_script_bytes=16 * 1024 * 1024,
        )

    def _stub(self, name: str, text: str) -> Path:
        path = self.bin / name
        path.write_text(text)
        path.chmod(0o700)
        return path

    def task(self, key: str, mode: str = "read", *extra: str, **options: Any) -> Task:
        return Task(key, (str(self.worker), str(self.out / key), mode, *extra), **options)

    def script(self, tasks: tuple[Task, ...], gpus: int = 0) -> Path:
        resources = Resources(cpus=1, memory_mib=100, time_limit="00:10:00", gpus=gpus)
        path = self.root / "batch.sh"
        path.write_bytes(render_batch(self.target, resources, tasks, ALLOCATION))
        return path

    def run(self, shell: str, script: Path, **env: str) -> subprocess.CompletedProcess[bytes]:
        # Slurm's --export=NIL: the batch starts with (almost) nothing but SLURM_* variables.
        environment = {"SLURM_JOB_ID": "4242", **env}
        return subprocess.run(
            [shell, str(script)],
            env={key: value for key, value in environment.items() if value},
            cwd=self.root,
            capture_output=True,
            timeout=60,
            check=False,
        )

    def record(self, key: str) -> dict[str, Any]:
        return json.loads((self.out / f"{key}.json").read_text())

    def started(self, key: str) -> bool:
        return (self.out / f"{key}.started").exists()


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
    result = runtime.run(shell, runtime.script(tasks, gpus), SLURM_RESTART_COUNT="2")
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "never").exists()
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
            SERVATUS_JOB_ID="4242",
            SERVATUS_RESTART_COUNT="2",
        )
        assert record["env"] == expected
        step = json.loads((runtime.records / f"servatus-{ALLOCATION}-{slot}.srun.json").read_text())
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
        assert ("--gres=gpu:a100:1" in options) == bool(gpus)
        if container:
            launch = json.loads((runtime.records / f"apptainer-{slot}.json").read_text())
            work = str(tmp_path / "work")
            flags = ["--cleanenv", "--bind", f"{work}:{work}", "--bind", "/data:/data:ro"]
            flags += ["--pwd", work] + (["--nv"] if gpus else [])
            assert launch == {"flags": flags, "image": "/images/work.sif"}


def test_restart_count_defaults_to_zero(tmp_path: Path) -> None:
    runtime = Runtime(tmp_path, container=False)
    result = runtime.run("/bin/sh", runtime.script((runtime.task("t"),)))
    assert result.returncode == 0, result.stderr
    assert runtime.record("t")["env"]["SERVATUS_RESTART_COUNT"] == "0"


@pytest.mark.parametrize("container", [False, True], ids=["direct", "apptainer"])
def test_missing_job_id_fails_before_any_step(tmp_path: Path, container: bool) -> None:
    runtime = Runtime(tmp_path, container=container)
    result = runtime.run("/bin/sh", runtime.script((runtime.task("t"),)), SLURM_JOB_ID="")
    assert result.returncode != 0
    assert b"Slurm did not set SLURM_JOB_ID" in result.stderr
    assert not runtime.started("t")
    assert list(runtime.records.iterdir()) == []


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


def test_worker_may_close_large_stdin_early(tmp_path: Path) -> None:
    runtime = Runtime(tmp_path, container=False)
    tasks = (runtime.task("early", "early", stdin=b"x" * 2_000_000),)
    result = runtime.run("/bin/sh", runtime.script(tasks))
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("shell", SHELLS)
def test_interrupt_kills_and_reaps_every_recorded_step(tmp_path: Path, shell: str) -> None:
    runtime = Runtime(tmp_path, container=False)
    keys = ("b0", "b1", "b2")
    script = runtime.script(tuple(runtime.task(key, "block", stdin=b"x") for key in keys))
    with subprocess.Popen(
        [shell, str(script)], env={"SLURM_JOB_ID": "4242"}, cwd=tmp_path, stderr=subprocess.PIPE
    ) as child:
        try:
            deadline = time.monotonic() + 20
            while not all(runtime.started(key) for key in keys):
                assert time.monotonic() < deadline, "steps never started"
                time.sleep(0.02)
            child.send_signal(signal.SIGTERM)
            assert child.wait(timeout=20) != 0
        finally:
            if child.poll() is None:
                child.kill()
    assert all((runtime.out / f"{key}.stopped").exists() for key in keys)


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
