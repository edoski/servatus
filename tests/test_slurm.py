from __future__ import annotations

import base64
from pathlib import Path

import pytest
from test_campaign import profile, resources, target, tasks

from servatus import Campaign, JobReceipt, ReconciliationError, ResourceRequest, Task, _slurm


def script_for(tmp_path: Path, request: ResourceRequest) -> str:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    return (
        campaign.plan(
            profile(
                target(
                    gpu_gres=None if request.gpus_per_task == 0 else "gpu:a100",
                    max_gpus_per_allocation=request.gpus_per_task,
                ),
                request,
            ),
            view=campaign.inspect(scheduler=False),
        )
        ._allocations[0]
        .script.decode()
    )


def test_cpu_script_omits_gpu_and_forbidden_flags(tmp_path: Path) -> None:
    script = script_for(tmp_path, resources(gpus_per_task=0))
    assert "--gres" not in script
    assert "--nv" not in script
    assert "#SBATCH" not in script
    assert "--overlap" not in script
    assert "--mem=0" not in script


def test_two_gpu_script_starts_one_exact_process(tmp_path: Path) -> None:
    script = script_for(tmp_path, resources(gpus_per_task=2))
    assert "--gres=gpu:a100:2" in script
    assert "--nv" in script
    assert script.count("--ntasks=1") == 1
    assert "--exclusive --exact --nodes=1" in script
    assert "/usr/bin/apptainer run --cleanenv" in script


def test_multitask_script_starts_all_siblings_before_waiting(tmp_path: Path) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(4))
    plan = campaign.plan(profile(), view=campaign.inspect(scheduler=False))
    script = plan._allocations[0].script.decode()
    assert script.count("/opt/slurm/bin/srun --exclusive") == 4
    assert script.index("pid_4=$!") < script.index('wait "$pid_1"')
    assert 'exit "$status"' in script


def test_job_id_logs_preserve_zero_based_combined_allocation_and_slot_shape(
    tmp_path: Path,
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(2))
    plan = campaign.plan(profile(), view=campaign.inspect(scheduler=False))
    argv = plan._allocations[0].argv
    allocation_id = plan._allocations[0].allocation_id
    script = plan._allocations[0].script.decode()

    assert f"--output=/cluster/logs/project/{allocation_id}-%j.out" in argv
    assert f"--error=/cluster/logs/project/{allocation_id}-%j.out" in argv
    for slot in range(2):
        path = f"/cluster/logs/project/{allocation_id}-%j-{slot}.out"
        assert f"--output={path}" in script
        assert f"--error={path}" in script
    assert "%j-2.out" not in script
    assert ".err" not in script


def test_binary_payload_is_embedded_before_acceptance_without_raw_bytes(tmp_path: Path) -> None:
    payload = b"line one\n\x00\xffline two"
    campaign = Campaign.open(tmp_path / "campaign", (Task("binary", ("run",), payload),))
    plan = campaign.plan(profile(), view=campaign.inspect(scheduler=False))
    script = plan._allocations[0].script
    assert base64.b64encode(payload) in script
    assert payload not in script


def test_local_ssh_environment_drops_scheduler_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SBATCH_PARTITION", "hostile")
    monkeypatch.setenv("SLURM_CONF", "/hostile")
    monkeypatch.setenv("UNRELATED", "hostile")
    environment = _slurm._ssh_environment()
    assert "SBATCH_PARTITION" not in environment
    assert "SLURM_CONF" not in environment
    assert "UNRELATED" not in environment
    assert environment["PATH"] == "/usr/bin:/bin"
    assert set(environment) <= {
        "PATH",
        "LANG",
        "LC_ALL",
        "HOME",
        "LOGNAME",
        "SSH_AUTH_SOCK",
        "USER",
    }


@pytest.mark.parametrize(
    ("output", "expected"),
    [(b"42\n", (42, None)), (b"42;cluster-a\n", (42, "cluster-a"))],
)
def test_parse_receipt(output: bytes, expected: tuple[int, str | None]) -> None:
    assert _slurm.parse_receipt(output) == expected


def test_receipt_string_is_slurm_identity() -> None:
    assert str(JobReceipt("allocation", 42, None, ("task",))) == "42"
    assert str(JobReceipt("allocation", 42, "alpha", ("task",))) == "42;alpha"


@pytest.mark.parametrize("output", [b"0\n", b"-1\n", b"42;bad;extra\n", b"noise 42\n", b""])
def test_parse_receipt_rejects_unproved_identity(output: bytes) -> None:
    with pytest.raises(ValueError):
        _slurm.parse_receipt(output)


def test_identity_query_accepts_absent_accounting_comment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def query(*args: object, **kwargs: object) -> _slurm.Result:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _slurm.Result(0, b"42|servatus-abc|servatus-abc\n", b"")
        return _slurm.Result(0, b"42|servatus-abc||alpha\n", b"")

    monkeypatch.setattr(_slurm, "_run_ssh", query)
    match = _slurm.query_identity(
        target(),
        job_name="servatus-abc",
        window_start="2030-01-01T00:00:00",
        window_end="2030-01-01T02:00:00",
    )
    assert match == _slurm.IdentityMatch(42, "alpha")
    assert calls == 2


@pytest.mark.parametrize(
    ("squeue", "sacct"),
    [
        (b"", b""),
        (
            b"42|servatus-abc|servatus-abc\n",
            b"43|servatus-abc|servatus-abc|alpha\n",
        ),
        (b"42|wrong|wrong\n", b""),
        (
            b"42|servatus-abc|servatus-abc\n",
            b"42|servatus-abc|wrong-identity|alpha\n",
        ),
    ],
)
def test_identity_query_leaves_unproved_results_ambiguous(
    monkeypatch: pytest.MonkeyPatch, squeue: bytes, sacct: bytes
) -> None:
    outputs = iter((squeue, sacct))
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *args, **kwargs: _slurm.Result(0, next(outputs), b""),
    )
    with pytest.raises(ReconciliationError):
        _slurm.query_identity(
            target(),
            job_name="servatus-abc",
            window_start="2030-01-01T00:00:00",
            window_end="2030-01-01T02:00:00",
        )
