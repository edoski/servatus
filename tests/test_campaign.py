from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

from servatus import (
    AmbiguousSubmission,
    Campaign,
    ConfigurationError,
    PlanError,
    ResourceRequest,
    SlurmTarget,
    Task,
    TaskConflict,
    _campaign,
    _slurm,
)


def target(**changes: object) -> SlurmTarget:
    values: dict[str, object] = {
        "host": "login.example.edu",
        "slurm_bin": PurePosixPath("/opt/slurm/bin"),
        "apptainer": PurePosixPath("/usr/bin/apptainer"),
        "image": PurePosixPath("/cluster/images/work.sif"),
        "work_root": PurePosixPath("/cluster/work/project"),
        "log_root": PurePosixPath("/cluster/logs/project"),
        "partitions": ("gpu",),
        "account": "research",
        "qos": None,
        "constraint": None,
        "gpu_gres": "gpu:a100",
        "max_tasks_per_allocation": 4,
        "max_cpus_per_allocation": 128,
        "max_memory_mib_per_allocation": 262144,
        "max_gpus_per_allocation": 4,
        "max_time_limit": "7-00:00:00",
        "max_allocations_per_submit": 64,
        "max_script_bytes": 4_194_304,
    }
    values.update(changes)
    return SlurmTarget(**values)  # type: ignore[arg-type]


def resources(**changes: object) -> ResourceRequest:
    values: dict[str, object] = {
        "cpus_per_task": 32,
        "memory_mib_per_task": 65536,
        "gpus_per_task": 1,
        "time_limit": "3-00:00:00",
    }
    values.update(changes)
    return ResourceRequest(**values)  # type: ignore[arg-type]


def tasks(count: int) -> tuple[Task, ...]:
    return tuple(
        Task(f"task-{index}", ("run", str(index)), f"input-{index}".encode())
        for index in range(count)
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cpus_per_task", True),
        ("cpus_per_task", 0),
        ("memory_mib_per_task", 0),
        ("gpus_per_task", -1),
        ("time_limit", "00:00:00"),
        ("time_limit", "UNLIMITED"),
    ],
)
def test_resource_request_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ConfigurationError):
        resources(**{field: value})


def test_strict_toml_rejects_unknown_and_counted_gres(tmp_path: Path) -> None:
    resource_file = tmp_path / "resources.toml"
    resource_file.write_text(
        "cpus_per_task = 1\nmemory_mib_per_task = 1024\ngpus_per_task = 0\n"
        'time_limit = "00:10:00"\nextra = true\n'
    )
    with pytest.raises(ConfigurationError, match="unknown"):
        ResourceRequest.from_toml(resource_file)

    with pytest.raises(ConfigurationError, match="count"):
        target(gpu_gres="gpu:a100:2")
    with pytest.raises(ConfigurationError):
        target(host="-oProxyCommand=bad")
    with pytest.raises(ConfigurationError):
        target(image=PurePosixPath("relative.sif"))


def test_campaign_freezes_task_order_and_payload(tmp_path: Path) -> None:
    path = tmp_path / "campaign"
    original = tasks(2)
    Campaign.open(path, original)

    with pytest.raises(TaskConflict):
        Campaign.open(path, tuple(reversed(original)))
    with pytest.raises(TaskConflict):
        Campaign.open(path, (replace(original[0], stdin=b"changed"), original[1]))


def test_kairos_shape_and_balanced_order(tmp_path: Path) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(9))
    plan = campaign.plan(target(), resources(), tasks_per_allocation=3)

    assert [len(allocation.task_keys) for allocation in plan.allocations] == [3, 3, 3]
    assert tuple(key for allocation in plan.allocations for key in allocation.task_keys) == tuple(
        task.key for task in tasks(9)
    )
    assert all(
        (allocation.cpus, allocation.memory_mib, allocation.gpus, allocation.time_limit)
        == (96, 196608, 3, "3-00:00:00")
        for allocation in plan.allocations
    )

    four = Campaign.open(tmp_path / "four", tasks(4)).plan(target(), resources())
    assert four.allocations[0].cpus == 128
    assert four.allocations[0].memory_mib == 262144
    assert four.allocations[0].gpus == 4
    argv = four._allocations[0].argv
    assert "--nodes=1" in argv
    assert "--ntasks=4" in argv
    assert "--cpus-per-task=32" in argv
    assert "--mem=262144M" in argv
    assert "--gres=gpu:a100:4" in argv
    assert "--time=3-00:00:00" in argv
    assert "--exclusive" not in argv
    script = four._allocations[0].script.decode()
    assert script.count("--cpus-per-task=32") == 4
    assert script.count("--mem=65536M") == 4
    assert script.count("--gres=gpu:a100:1") == 4


def test_empty_campaign_has_empty_plan(tmp_path: Path) -> None:
    plan = Campaign.open(tmp_path / "campaign", ()).plan(target(), resources())
    assert plan.allocations == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"max_cpus_per_allocation": 31},
        {"max_memory_mib_per_allocation": 65535},
        {"max_gpus_per_allocation": 0, "gpu_gres": None},
        {"max_time_limit": "2-23:59:59"},
    ],
)
def test_every_target_ceiling_can_reject_one_task(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    with pytest.raises(PlanError):
        campaign.plan(target(**changes), resources())


@pytest.mark.parametrize("count", range(1, 18))
@pytest.mark.parametrize("gpus", [0, 1, 2, 4])
def test_balanced_groups_preserve_exact_resources(tmp_path: Path, count: int, gpus: int) -> None:
    maximum = 4
    gpu_ceiling = 0 if gpus == 0 else maximum * gpus
    plan = Campaign.open(tmp_path / f"campaign-{count}-{gpus}", tasks(count)).plan(
        target(
            gpu_gres=None if gpus == 0 else "gpu",
            max_gpus_per_allocation=gpu_ceiling,
            max_cpus_per_allocation=maximum * 2,
            max_memory_mib_per_allocation=maximum * 100,
        ),
        resources(
            cpus_per_task=2,
            memory_mib_per_task=100,
            gpus_per_task=gpus,
            time_limit="00:10:00",
        ),
    )
    sizes = [len(allocation.task_keys) for allocation in plan.allocations]
    assert sizes == sorted(sizes, reverse=True)
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == count
    assert all(allocation.cpus == len(allocation.task_keys) * 2 for allocation in plan.allocations)
    assert all(
        allocation.memory_mib == len(allocation.task_keys) * 100 for allocation in plan.allocations
    )
    assert all(
        allocation.gpus == len(allocation.task_keys) * gpus for allocation in plan.allocations
    )
    assert all(allocation.time_limit == "00:10:00" for allocation in plan.allocations)


def test_requested_cap_above_feasible_is_rejected(tmp_path: Path) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(4))
    with pytest.raises(PlanError, match="capacity"):
        campaign.plan(target(max_tasks_per_allocation=2), resources(), tasks_per_allocation=3)


def test_plan_is_local_stable_and_public_document_redacts_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(
        tmp_path / "campaign",
        (Task("binary", ("value with spaces", "line\nbreak"), b"\x00\xff\nopaque"),),
    )
    before = (tmp_path / "campaign" / "campaign.json").read_bytes()

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("planning contacted an external process")

    monkeypatch.setattr(_slurm, "_run_ssh", forbidden)
    first = campaign.plan(target(), resources())
    second = campaign.plan(target(), resources())
    document = _campaign.plan_document(first)

    assert first.digest == second.digest
    assert before == (tmp_path / "campaign" / "campaign.json").read_bytes()
    encoded = json.dumps(document)
    assert "opaque" not in encoded
    assert "value with spaces" not in encoded


def test_script_size_exact_boundary(tmp_path: Path) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    roomy = campaign.plan(target(), resources())
    size = len(roomy._allocations[0].script)
    exact_target = replace(target(), max_script_bytes=size)

    campaign.plan(exact_target, resources())
    with pytest.raises(PlanError, match="script"):
        campaign.plan(replace(exact_target, max_script_bytes=size - 1), resources())


def test_submit_records_intent_before_ssh_and_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    plan = campaign.plan(target(), resources())

    def accepted(target_value: SlurmTarget, argv: tuple[str, ...], script: bytes) -> _slurm.Result:
        state = json.loads((tmp_path / "campaign" / "campaign.json").read_text())
        assert len(state["intents"]) == 1
        assert state["receipts"] == []
        assert script == plan._allocations[0].script
        return _slurm.Result(0, b"4242;alpha\n", b"")

    monkeypatch.setattr(_slurm, "_run_ssh", accepted)
    receipts = campaign.submit(plan)

    assert [(receipt.job_id, receipt.cluster) for receipt in receipts] == [(4242, "alpha")]
    assert campaign.status().pending_task_keys == ()


def test_accepted_without_receipt_is_ambiguous_and_halts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(5))
    plan = campaign.plan(replace(target(), max_tasks_per_allocation=2), resources())
    calls = 0

    def accepted(target_value: SlurmTarget, argv: tuple[str, ...], script: bytes) -> _slurm.Result:
        nonlocal calls
        calls += 1
        return _slurm.Result(0, b"88\n", b"")

    def fail_receipt(*args: object, **kwargs: object) -> None:
        raise OSError("injected receipt write failure")

    monkeypatch.setattr(_slurm, "_run_ssh", accepted)
    monkeypatch.setattr(campaign, "_record_receipt", fail_receipt)

    with pytest.raises(AmbiguousSubmission):
        campaign.submit(plan)
    assert calls == 1
    assert len(campaign.status().ambiguous_allocation_ids) == 1


def test_manual_resolution_and_explicit_retry_preserve_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    plan = campaign.plan(target(), resources())
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *args, **kwargs: _slurm.Result(0, b"not-a-job\n", b""),
    )
    with pytest.raises(AmbiguousSubmission):
        campaign.submit(plan)

    allocation_id = campaign.status().ambiguous_allocation_ids[0]
    campaign.resolve(allocation_id, job_id=None)
    retry_plan = campaign.plan(target(), resources())
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *args, **kwargs: _slurm.Result(0, b"101\n", b""),
    )
    campaign.submit(retry_plan)
    second_retry = campaign.plan(target(), resources(), retry={"task-0"})
    campaign.submit(second_retry)
    assert [receipt.job_id for receipt in campaign.status().receipts] == [101, 101]


def test_stale_foreign_and_tampered_plans_fail(tmp_path: Path) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    plan = campaign.plan(target(), resources())
    foreign = Campaign.open(tmp_path / "foreign", tasks(1))
    with pytest.raises(PlanError):
        foreign.submit(plan)

    object.__setattr__(plan, "_digest", "0" * 64)
    with pytest.raises(PlanError):
        campaign.submit(plan)


def test_submission_call_cap_leaves_later_groups_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(5))
    limited = target(max_tasks_per_allocation=2, max_allocations_per_submit=2)
    plan = campaign.plan(limited, resources())
    calls = 0

    def accepted(*args: object, **kwargs: object) -> _slurm.Result:
        nonlocal calls
        calls += 1
        return _slurm.Result(0, f"{100 + calls}\n".encode(), b"")

    monkeypatch.setattr(_slurm, "_run_ssh", accepted)
    receipts = campaign.submit(plan)

    assert len(receipts) == 2
    assert calls == 2
    assert campaign.status().pending_task_keys == ("task-4",)
    with pytest.raises(PlanError, match="stale"):
        campaign.submit(plan)


def test_validate_deduplicates_shapes_and_never_mutates_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(5))
    plan = campaign.plan(target(max_tasks_per_allocation=2), resources())
    before = (tmp_path / "campaign" / "campaign.json").read_bytes()
    calls: list[tuple[str, ...]] = []

    def validated(target_value: SlurmTarget, argv: tuple[str, ...], script: bytes) -> _slurm.Result:
        calls.append(argv)
        return _slurm.Result(0, b"Job 1 to start at 2030-01-01\n", b"")

    monkeypatch.setattr(_slurm, "_run_ssh", validated)
    results = _campaign.validate_plan(plan)

    assert len(results) == 2  # group shapes 2 and 1
    assert all(argv[-1] == "--test-only" for argv in calls)
    assert before == (tmp_path / "campaign" / "campaign.json").read_bytes()


def test_reconcile_adopts_only_private_query_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    plan = campaign.plan(target(), resources())
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *args, **kwargs: _slurm.Result(1, b"", b"lost reply"),
    )
    with pytest.raises(AmbiguousSubmission):
        campaign.submit(plan)
    allocation_id = campaign.status().ambiguous_allocation_ids[0]
    monkeypatch.setattr(
        _slurm,
        "query_identity",
        lambda *args, **kwargs: _slurm.IdentityMatch(909, "alpha"),
    )

    receipt = campaign.reconcile(target(), allocation_id)

    assert (receipt.job_id, receipt.cluster) == (909, "alpha")
    assert campaign.status().ambiguous_allocation_ids == ()
