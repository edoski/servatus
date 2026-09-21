from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

import pytest
from test_campaign import resources, target

from servatus import ObservationError, SlurmTarget, Task, _slurm


def synthetic_runtime(tmp_path: Path) -> tuple[SlurmTarget, Path]:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    srun = tmp_path / "srun"
    srun.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "args = sys.argv[1:]\n"
        "while args and args[0].startswith('--'): args.pop(0)\n"
        "assert os.environ['SLURM_EXPORT_ENV'] == 'ALL'\n"
        "if os.environ.get('NO_GPU'): os.environ.pop('CUDA_VISIBLE_DEVICES', None)\n"
        "else: os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU-step-2,GPU-step-7'\n"
        "os.execv(args[0], args)\n"
    )
    srun.chmod(0o700)
    apptainer = tmp_path / "apptainer"
    apptainer.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, signal, sys, time\n"
        "args = sys.argv[1:]\n"
        "assert args.pop(0) == 'run'\n"
        "env = {}\n"
        "while args[0].startswith('--'):\n"
        "    flag = args.pop(0)\n"
        "    if flag in ('--bind', '--pwd'): args.pop(0)\n"
        "    elif flag == '--env':\n"
        "        key, value = args.pop(0).split('=', 1); env[key] = value\n"
        "image = args.pop(0)\n"
        "out = pathlib.Path(args[0])\n"
        "mode = args[1]\n"
        "def stop(*_):\n"
        "    out.with_suffix('.stopped').touch(); raise SystemExit(1)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "out.with_suffix('.started').touch()\n"
        "if mode == 'block': time.sleep(60)\n"
        "if mode == 'slow': time.sleep(0.1)\n"
        "assert any(pathlib.Path(os.environ['TMPDIR']).iterdir())\n"
        "data = b'' if mode == 'early' else sys.stdin.buffer.read()\n"
        "out.write_bytes(data)\n"
        "out.with_suffix('.json').write_text(json.dumps([args, env]))\n"
        "raise SystemExit(7 if mode == 'fail' else 0)\n"
    )
    apptainer.chmod(0o700)
    return target(slurm_bin=PurePosixPath(tmp_path), apptainer=PurePosixPath(apptainer)), scratch


def run_script(script: bytes, scratch: Path, **env: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["/bin/sh"],
        input=script,
        capture_output=True,
        timeout=10,
        check=False,
        env={**os.environ, "TMPDIR": str(scratch), **env},
    )


@pytest.mark.parametrize("gpus", [0, 2])
def test_generated_steps_preserve_binary_input_argv_and_step_gpu_selection(
    tmp_path: Path, gpus: int
) -> None:
    route, scratch = synthetic_runtime(tmp_path)
    output = tmp_path / "payload"
    args = (str(output), "read", "a b", "'quoted'", '"double"', "$(touch not-executed)", "", "λ")
    payload = bytes(range(256)) * 4096
    script = _slurm.render_script(
        route, resources(gpus_per_task=gpus), (Task("t", args, payload),), "abc"
    )
    result = run_script(script, scratch, CUDA_VISIBLE_DEVICES="allocation-wide")
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == payload
    actual_args, environment = json.loads(output.with_suffix(".json").read_text())
    assert actual_args == list(args)
    assert environment == (
        {"CUDA_VISIBLE_DEVICES": "GPU-step-2,GPU-step-7", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        if gpus
        else {}
    )
    assert list(scratch.iterdir()) == []


def test_successful_worker_may_close_large_input_early(tmp_path: Path) -> None:
    route, scratch = synthetic_runtime(tmp_path)
    task = Task("early", (str(tmp_path / "out"), "early"), b"x" * 2_000_000)
    result = run_script(_slurm.render_script(route, resources(), (task,), "abc"), scratch)
    assert result.returncode == 0, result.stderr
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("failure", ["decoder", "scratch", "gpu"])
def test_setup_failure_cannot_launch_a_successful_worker(tmp_path: Path, failure: str) -> None:
    route, scratch = synthetic_runtime(tmp_path)
    output = tmp_path / "out"
    script = _slurm.render_script(
        route, resources(), (Task("t", (str(output), "read"), b"x"),), "abc"
    )
    if failure == "decoder":
        decoder = tmp_path / "decoder"
        decoder.write_text("#!/bin/sh\nprintf partial\nexit 7\n")
        decoder.chmod(0o700)
        script = script.replace(b"/usr/bin/base64", str(decoder).encode())
    environment = {"NO_GPU": "1"} if failure == "gpu" else {}
    result = run_script(
        script, scratch / "absent" if failure == "scratch" else scratch, **environment
    )
    assert result.returncode != 0
    assert not output.with_suffix(".started").exists()
    assert list(scratch.iterdir()) == []


def test_failure_waits_for_started_siblings_before_removing_payloads(tmp_path: Path) -> None:
    route, scratch = synthetic_runtime(tmp_path)
    tasks = tuple(
        Task(mode, (str(tmp_path / mode), mode), mode.encode()) for mode in ("fail", "slow")
    )
    result = run_script(_slurm.render_script(route, resources(), tasks, "abc"), scratch)
    assert result.returncode != 0
    assert (tmp_path / "slow").read_bytes() == b"slow"
    assert list(scratch.iterdir()) == []


def test_interrupt_waits_for_children_and_cleans_private_payloads(tmp_path: Path) -> None:
    route, scratch = synthetic_runtime(tmp_path)
    outputs = tuple(tmp_path / str(index) for index in range(2))
    tasks = tuple(
        Task(str(index), (str(output), "block"), b"x") for index, output in enumerate(outputs)
    )
    script = tmp_path / "batch.sh"
    script.write_bytes(_slurm.render_script(route, resources(), tasks, "abc"))
    with subprocess.Popen(
        ["/bin/sh", str(script)], env={**os.environ, "TMPDIR": str(scratch)}
    ) as child:
        try:
            deadline = time.monotonic() + 5
            while not all(output.with_suffix(".started").exists() for output in outputs):
                assert time.monotonic() < deadline
                time.sleep(0.01)
            child.send_signal(signal.SIGTERM)
            assert child.wait(timeout=5) != 0
        finally:
            if child.poll() is None:
                child.kill()
    assert all(output.with_suffix(".stopped").exists() for output in outputs)
    assert list(scratch.iterdir()) == []


def test_submission_transport_streams_binary_input_and_drains_both_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_popen = subprocess.Popen
    payload = bytes(range(256)) * 8192

    def local(*_args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        return real_popen(
            [
                sys.executable,
                "-c",
                "import sys; data=sys.stdin.buffer.read(); "
                "sys.stderr.buffer.write(b'e'*200000); sys.stdout.buffer.write(data)",
            ],
            **kwargs,
        )

    monkeypatch.setattr(_slurm.subprocess, "Popen", local)
    result = _slurm._run_bounded_ssh(
        target(), ("/test",), stdin=payload, max_stdout_bytes=len(payload)
    )
    assert result == _slurm.Result(0, payload, b"e" * 200000)


@pytest.mark.parametrize("failure", ["selector", "timeout", "overflow"])
def test_transport_owns_child_and_pipes_from_spawn_through_failure(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    real_popen = subprocess.Popen
    children: list[subprocess.Popen[bytes]] = []

    def local(*_args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        program = "import time; time.sleep(60)"
        if failure == "overflow":
            program = "import os, time; os.write(2, b'x' * 1048577); time.sleep(60)"
        child = real_popen([sys.executable, "-c", program], **kwargs)
        children.append(child)
        return child

    def broken_selector() -> None:
        raise OSError("selector unavailable")

    monkeypatch.setattr(_slurm.subprocess, "Popen", local)
    if failure == "selector":
        monkeypatch.setattr(_slurm.selectors, "DefaultSelector", broken_selector)
    if failure == "timeout":
        monkeypatch.setattr(_slurm, "_SSH_TIMEOUT_SECONDS", 0.02)
    with pytest.raises((OSError, subprocess.TimeoutExpired, ObservationError)):
        _slurm._run_ssh(target(), ("/test",), b"binary\x00input")
    assert children[0].poll() is not None
    assert children[0].stdout.closed and children[0].stderr.closed


def test_interrupt_between_launch_and_pid_record_still_reaps_the_child(tmp_path: Path) -> None:
    route, scratch = synthetic_runtime(tmp_path)
    output = tmp_path / "out"
    task = Task("slow", (str(output), "slow"), b"x")
    script = _slurm.render_script(route, resources(), (task,), "abc")
    script = script.replace(b"pid_1=$!", b"kill -TERM $$\npid_1=$!")
    result = run_script(script, scratch)
    assert result.returncode != 0
    if output.with_suffix(".started").exists():
        assert output.read_bytes() == b"x"
    assert list(scratch.iterdir()) == []
