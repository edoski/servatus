from __future__ import annotations

import base64
import copy
import json
import os
import stat
import threading
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pytest

from servatus import (
    AmbiguousSubmission,
    Campaign,
    ConfigurationError,
    PlanError,
    Profile,
    ResourceRequest,
    SlurmTarget,
    Task,
    TaskConflict,
    ValidationResult,
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


def profile(
    target_value: SlurmTarget | None = None,
    resource_value: ResourceRequest | None = None,
    *,
    label: str = "test",
) -> Profile:
    return Profile(label, target_value or target(), resource_value or resources())


def planning(
    campaign: Campaign,
    target_value: SlurmTarget | None = None,
    resource_value: ResourceRequest | None = None,
    **options: object,
) -> _campaign.SubmissionPlan:
    return campaign.plan(
        profile(target_value, resource_value),
        view=campaign.inspect(),
        **options,  # type: ignore[arg-type]
    )


@pytest.fixture(autouse=True)
def terminal_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    def observe(
        _target: SlurmTarget, queries: tuple[_slurm._AttemptQuery, ...]
    ) -> tuple[_slurm.SchedulerObservation, ...]:
        return tuple(
            _slurm.SchedulerObservation(
                _slurm.AllocationState.SUCCEEDED,
                "COMPLETED",
                "COMPLETED",
                "0:0",
                None,
                datetime.now(UTC).isoformat(),
                datetime.now(UTC).isoformat(),
            )
            for _query in queries
        )

    monkeypatch.setattr(_slurm, "query_attempts", observe)


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


def test_direct_target_and_resources_reject_invalid_values() -> None:
    with pytest.raises(ConfigurationError):
        target(host="-oProxyCommand=bad")
    with pytest.raises(ConfigurationError):
        target(image=PurePosixPath("relative.sif"))


@pytest.mark.parametrize(
    "gres",
    ["scratch", "gpu:2", "gpu:a100:2", "gpu:a100/bad", "gpu:a100\n"],
)
def test_gpu_gres_accepts_only_count_free_gpu_family(gres: str) -> None:
    with pytest.raises(ConfigurationError):
        target(gpu_gres=gres)


def test_direct_target_normalizes_pathlike_values_and_rejects_other_types() -> None:
    direct = target(slurm_bin="/opt/slurm/bin", image=Path("/images/work.sif"))
    assert direct.slurm_bin == PurePosixPath("/opt/slurm/bin")
    assert direct.image == PurePosixPath("/images/work.sif")
    assert isinstance(direct.slurm_bin, PurePosixPath)
    with pytest.raises(ConfigurationError):
        target(slurm_bin=7)


def test_campaign_accepts_only_an_exact_append_only_suffix(tmp_path: Path) -> None:
    path = tmp_path / "campaign"
    original = tasks(2)
    campaign = Campaign.open(path, original)
    stale = planning(campaign)
    before = json.loads((path / "campaign.json").read_text())

    grown = Campaign.open(path, tasks(4))
    after = json.loads((path / "campaign.json").read_text())

    assert after["revision"] == before["revision"] + 1
    assert after["tasks"][:2] == before["tasks"]
    assert after["tasks"] == [_campaign._task_record(task) for task in tasks(4)]
    unchanged = (path / "campaign.json").read_bytes()
    Campaign.open(path, tasks(4))
    assert (path / "campaign.json").read_bytes() == unchanged
    with pytest.raises(PlanError, match="stale"):
        grown.submit(stale)

    with pytest.raises(TaskConflict):
        Campaign.open(path, original)
    with pytest.raises(TaskConflict):
        Campaign.open(path, tuple(reversed(tasks(4))))
    with pytest.raises(TaskConflict):
        Campaign.open(path, (replace(original[0], stdin=b"changed"), *tasks(4)[1:]))


def test_campaign_loads_an_existing_durable_roster(tmp_path: Path) -> None:
    path = tmp_path / "campaign"
    opened = Campaign.open(path, tasks(2))

    loaded = Campaign.load(path)

    assert opened.tasks == tasks(2)
    assert loaded.tasks == tasks(2)


def test_campaign_seal_is_durable_idempotent_and_stales_plans(tmp_path: Path) -> None:
    path = tmp_path / "campaign"
    campaign = Campaign.open(path, tasks(2))
    stale = planning(campaign)

    campaign.seal()
    sealed = (path / "campaign.json").read_bytes()
    loaded = Campaign.load(path)

    assert loaded.tasks == tasks(2)
    loaded.seal()
    assert (path / "campaign.json").read_bytes() == sealed
    Campaign.open(path, tasks(2))
    with pytest.raises(PlanError, match="stale"):
        loaded.validate(stale)


@pytest.mark.parametrize(
    "changed",
    [
        tasks(3),
        tasks(1),
        tuple(reversed(tasks(2))),
        (replace(tasks(2)[0], stdin=b"changed"), tasks(2)[1]),
    ],
)
def test_sealed_campaign_rejects_every_roster_change(
    tmp_path: Path, changed: tuple[Task, ...]
) -> None:
    path = tmp_path / "campaign"
    Campaign.open(path, tasks(2)).seal()

    with pytest.raises(TaskConflict):
        Campaign.open(path, changed)


def test_open_campaign_can_execute_before_sealing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, b"101\n", b""),
    )

    assert campaign.submit(planning(campaign))[0].job_id == 101


def test_growth_preserves_receipts_and_submits_only_new_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "campaign"
    campaign = Campaign.open(path, tasks(2))
    submitted: list[tuple[str, ...]] = []

    def accepted(_target: SlurmTarget, argv: tuple[str, ...], _script: bytes) -> _slurm.Result:
        submitted.append(argv)
        return _slurm.Result(0, f"{100 + len(submitted)}\n".encode(), b"")

    monkeypatch.setattr(_slurm, "_run_ssh", accepted)
    campaign.submit(planning(campaign))
    grown = Campaign.open(path, tasks(4))
    suffix_plan = planning(grown)

    assert [allocation.task_keys for allocation in suffix_plan.allocations] == [
        ("task-2", "task-3")
    ]
    grown.submit(suffix_plan)
    accepted_attempts = [
        attempt
        for attempt in grown.inspect(scheduler=False).attempts
        if attempt.receipt is not None
    ]
    assert [attempt.receipt.task_keys for attempt in accepted_attempts] == [  # type: ignore[union-attr]
        ("task-0", "task-1"),
        ("task-2", "task-3"),
    ]
    assert planning(grown).allocations == ()
    retry = planning(grown, retry={"task-0"})
    assert [allocation.task_keys for allocation in retry.allocations] == [("task-0",)]


def test_old_handle_reads_appended_roster_for_status_plan_and_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "campaign"
    old = Campaign.open(path, tasks(1))
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, b"101\n", b""),
    )
    old.submit(planning(old))
    Campaign.open(path, tasks(3))

    assert [allocation.task_keys for allocation in planning(old).allocations] == [
        ("task-1", "task-2")
    ]
    assert [allocation.task_keys for allocation in planning(old, retry={"task-0"}).allocations] == [
        ("task-0", "task-1", "task-2")
    ]


def test_append_after_submit_verification_blocks_before_ssh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "campaign"
    old = Campaign.open(path, tasks(1))
    plan = planning(old)
    verify = old._verify_plan
    ssh_calls = 0

    def verify_then_append(value: _campaign.SubmissionPlan, *, operation: str) -> None:
        verify(value, operation=operation)
        Campaign.open(path, tasks(2))

    def accepted(*_args: object, **_kwargs: object) -> _slurm.Result:
        nonlocal ssh_calls
        ssh_calls += 1
        return _slurm.Result(0, b"101\n", b"")

    monkeypatch.setattr(old, "_verify_plan", verify_then_append)
    monkeypatch.setattr(_slurm, "_run_ssh", accepted)

    with pytest.raises(PlanError, match="changed before submission freshness"):
        old.submit(plan)
    assert ssh_calls == 0
    assert old.tasks == tasks(2)


def test_growth_preserves_ambiguous_intent_and_resource_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "campaign"
    campaign = Campaign.open(path, tasks(1))

    def lose_reply(*_args: object, **_kwargs: object) -> _slurm.Result:
        raise OSError("lost")

    monkeypatch.setattr(_slurm, "_run_ssh", lose_reply)
    with pytest.raises(AmbiguousSubmission):
        campaign.submit(planning(campaign))
    ambiguous = tuple(
        attempt.allocation_id
        for attempt in campaign.inspect(scheduler=False).attempts
        if attempt.acceptance is _campaign.AcceptanceState.UNRESOLVED
    )

    grown = Campaign.open(path, tasks(2))

    assert (
        tuple(
            attempt.allocation_id
            for attempt in grown.inspect(scheduler=False).attempts
            if attempt.acceptance is _campaign.AcceptanceState.UNRESOLVED
        )
        == ambiguous
    )
    assert [allocation.task_keys for allocation in planning(grown).allocations] == [("task-1",)]
    campaign.resolve(ambiguous[0], job_id=None)
    with pytest.raises(PlanError, match="resource semantics"):
        planning(campaign, resource_value=resources(cpus_per_task=16))
    assert tuple(
        key for allocation in planning(campaign).allocations for key in allocation.task_keys
    ) == ("task-0", "task-1")


def test_growth_precommit_failure_preserves_prior_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "campaign"
    Campaign.open(path, tasks(1))
    before = (path / "campaign.json").read_bytes()
    real_write = _campaign.os.write

    def fail_write(_descriptor: int, _data: object) -> int:
        raise OSError("injected append write failure")

    monkeypatch.setattr(_campaign.os, "write", fail_write)
    with pytest.raises(OSError, match="append write failure"):
        Campaign.open(path, tasks(2))
    monkeypatch.setattr(_campaign.os, "write", real_write)

    assert (path / "campaign.json").read_bytes() == before
    Campaign.open(path, tasks(1))
    assert list(path.glob(".campaign-*.tmp")) == []


def test_growth_postcommit_sync_failure_is_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "campaign"
    Campaign.open(path, tasks(1))
    campaign_entry = path.stat()
    real_fsync = _campaign.os.fsync

    def fail_directory_sync(descriptor: int) -> None:
        entry = os.fstat(descriptor)
        if (entry.st_dev, entry.st_ino) == (campaign_entry.st_dev, campaign_entry.st_ino):
            raise OSError("injected append directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(_campaign.os, "fsync", fail_directory_sync)
    with pytest.raises(OSError, match="append directory fsync failure"):
        Campaign.open(path, tasks(2))
    monkeypatch.setattr(_campaign.os, "fsync", real_fsync)

    recovered = Campaign.open(path, tasks(2))
    state = json.loads((path / "campaign.json").read_text())
    assert state["revision"] == 1
    assert recovered.tasks == tasks(2)
    assert list(path.glob(".campaign-*.tmp")) == []


def test_growth_task_digest_tampering_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "campaign"
    Campaign.open(path, tasks(2))
    state_path = path / "campaign.json"
    state = json.loads(state_path.read_text())
    state["tasks"][1]["stdin"] = base64.b64encode(b"changed").decode()
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(TaskConflict, match="digest"):
        Campaign.open(path, tasks(2))


def test_kairos_shape_and_balanced_order(tmp_path: Path) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(9))
    plan = planning(campaign, tasks_per_allocation=3)

    assert [len(allocation.task_keys) for allocation in plan.allocations] == [3, 3, 3]
    assert tuple(key for allocation in plan.allocations for key in allocation.task_keys) == tuple(
        task.key for task in tasks(9)
    )
    assert all(
        (allocation.cpus, allocation.memory_mib, allocation.gpus, allocation.time_limit)
        == (96, 196608, 3, "3-00:00:00")
        for allocation in plan.allocations
    )

    four = planning(Campaign.open(tmp_path / "four", tasks(4)))
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
    plan = planning(Campaign.open(tmp_path / "campaign", ()))
    assert plan.allocations == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"max_cpus_per_allocation": 31},
        {"max_memory_mib_per_allocation": 65535},
        {"max_gpus_per_allocation": 0, "gpu_gres": None},
        {"max_time_limit": "2-23:58:59"},
    ],
)
def test_every_target_ceiling_can_reject_one_task(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    with pytest.raises(PlanError):
        planning(campaign, target(**changes))


@pytest.mark.parametrize("count", range(1, 18))
@pytest.mark.parametrize("gpus", [0, 1, 2, 4])
def test_balanced_groups_preserve_exact_resources(tmp_path: Path, count: int, gpus: int) -> None:
    maximum = 4
    gpu_ceiling = 0 if gpus == 0 else maximum * gpus
    campaign = Campaign.open(tmp_path / f"campaign-{count}-{gpus}", tasks(count))
    plan = planning(
        campaign,
        target(
            gpu_gres=None if gpus == 0 else "gpu",
            max_gpus_per_allocation=gpu_ceiling,
            max_cpus_per_allocation=maximum * 2,
            max_memory_mib_per_allocation=maximum * 100,
        ),
        resource_value=resources(
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
        planning(campaign, target(max_tasks_per_allocation=2), tasks_per_allocation=3)


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
    view = campaign.inspect()
    first = campaign.plan(profile(), view=view)
    second = campaign.plan(profile(), view=view)
    document = _campaign.plan_document(first)

    assert first.digest == second.digest
    assert before == (tmp_path / "campaign" / "campaign.json").read_bytes()
    encoded = json.dumps(document)
    assert "opaque" not in encoded
    assert "value with spaces" not in encoded


def test_effective_slurm_time_is_rounded_once_and_requested_time_is_preserved(
    tmp_path: Path,
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(4))
    plan = planning(
        campaign,
        target(max_time_limit="00:00:01"),
        resource_value=resources(time_limit="00:00:30"),
    )
    document = _campaign.plan_document(plan)
    assert document["resources"] == {
        "cpus_per_task": 32,
        "memory_mib_per_task": 65536,
        "gpus_per_task": 1,
        "time_limit": "00:00:30",
    }
    assert plan.allocations[0].time_limit == "00:01:00"
    assert "--time=00:01:00" in plan._allocations[0].argv

    seconds = planning(
        Campaign.open(tmp_path / "seconds", tasks(1)),
        resource_value=resources(time_limit="01:02:03"),
    )
    assert seconds.allocations[0].time_limit == "01:03:00"
    assert "--time=01:03:00" in seconds._allocations[0].argv

    with pytest.raises(PlanError, match="time_limit"):
        planning(
            Campaign.open(tmp_path / "too-long", tasks(1)),
            target(max_time_limit="00:01:00"),
            resource_value=resources(time_limit="00:01:01"),
        )


def test_script_size_exact_boundary(tmp_path: Path) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    roomy = planning(campaign)
    size = len(roomy._allocations[0].script)
    exact_target = replace(target(), max_script_bytes=size)

    planning(campaign, exact_target)
    with pytest.raises(PlanError, match="script"):
        planning(campaign, replace(exact_target, max_script_bytes=size - 1))


def test_submit_records_intent_before_ssh_and_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    plan = planning(campaign)

    def accepted(target_value: SlurmTarget, argv: tuple[str, ...], script: bytes) -> _slurm.Result:
        state = json.loads((tmp_path / "campaign" / "campaign.json").read_text())
        assert len(state["attempts"]) == 1
        assert state["attempts"][0]["acceptance"] == {"status": "UNRESOLVED"}
        assert state["lineage"]["target"] == _campaign._target_dict(target())
        assert state["lineage"]["resources"] == _campaign._resource_dict(resources())
        assert state["attempts"][0]["target_digest"] == state["lineage"]["target_digest"]
        assert state["attempts"][0]["resource_digest"] == state["lineage"]["resource_digest"]
        assert state["attempts"][0]["allocation"] == {
            "cpus": 32,
            "memory_mib": 65536,
            "gpus": 1,
            "time_limit": "3-00:00:00",
        }
        assert state["attempts"][0]["sbatch_argv"] == list(argv)
        assert script == plan._allocations[0].script
        return _slurm.Result(0, b"4242;alpha\n", b"")

    monkeypatch.setattr(_slurm, "_run_ssh", accepted)
    receipts = campaign.submit(plan)

    assert [(receipt.job_id, receipt.cluster) for receipt in receipts] == [(4242, "alpha")]
    assert planning(campaign).allocations == ()
    state_path = tmp_path / "campaign" / "campaign.json"
    assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
    assert list((tmp_path / "campaign").glob(".campaign-*.tmp")) == []


def test_submission_state_uses_only_authoritative_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "campaign"
    campaign = Campaign.open(path, tasks(1))
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, b"4242;alpha\n", b""),
    )

    campaign.submit(planning(campaign))
    state = json.loads((path / "campaign.json").read_text())

    assert state["schema_version"] == 4
    assert state["phase"] == "OPEN"
    assert set(state["attempts"][0]) == {
        "allocation_id",
        "task_keys",
        "campaign_revision",
        "profile_label",
        "retry_task_keys",
        "duplicate_risk_task_keys",
        "plan_digest",
        "script_digest",
        "target_digest",
        "resource_digest",
        "allocation",
        "sbatch_argv",
        "window_start",
        "window_end",
        "acceptance",
    }
    assert state["attempts"][0]["profile_label"] == "test"
    assert state["attempts"][0]["duplicate_risk_task_keys"] == []
    assert state["attempts"][0]["campaign_revision"] == 0
    assert state["attempts"][0]["retry_task_keys"] == []
    assert state["attempts"][0]["target_digest"] == state["lineage"]["target_digest"]
    assert state["attempts"][0]["resource_digest"] == state["lineage"]["resource_digest"]
    assert state["attempts"][0]["acceptance"] == {
        "status": "ACCEPTED",
        "job_id": 4242,
        "cluster": "alpha",
    }
    retry = planning(Campaign.open(path, tasks(1)), retry={"task-0"})
    allocation = _campaign.plan_document(retry)["allocations"][0]
    assert set(allocation) == {
        "allocation_id",
        "task_keys",
        "cpus",
        "memory_mib",
        "gpus",
        "time_limit",
        "sbatch_argv",
        "script_digest",
    }
    campaign.submit(retry)
    attempts = json.loads((path / "campaign.json").read_text())["attempts"]
    assert attempts[1]["campaign_revision"] == 2
    assert attempts[1]["retry_task_keys"] == ["task-0"]


def test_campaign_rejects_attempt_with_contradictory_acceptance_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "campaign"
    campaign = Campaign.open(path, tasks(1))
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(1, b"", b"lost reply"),
    )
    with pytest.raises(AmbiguousSubmission):
        campaign.submit(planning(campaign))

    state_path = path / "campaign.json"
    state = json.loads(state_path.read_text())
    state["attempts"][0]["acceptance"] = {
        "status": "ACCEPTED",
        "job_id": 42,
        "cluster": None,
        "not_submitted": True,
    }
    state_path.write_text(json.dumps(state))

    with pytest.raises(TaskConflict, match="acceptance"):
        Campaign.open(path, tasks(1))


@pytest.mark.parametrize("schema", [3, 3.0, True])
def test_campaign_rejects_old_or_noninteger_state_schema(tmp_path: Path, schema: object) -> None:
    path = tmp_path / "campaign"
    Campaign.open(path, tasks(1))
    state_path = path / "campaign.json"
    state = json.loads(state_path.read_text())
    state["schema_version"] = schema
    state_path.write_text(json.dumps(state))

    with pytest.raises(TaskConflict, match="schema"):
        Campaign.open(path, tasks(1))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("phase",), []),
        (("revision",), 2.0),
        (("tasks", 0, "key"), 7),
        (("attempts", 0, "campaign_revision"), True),
        (("attempts", 0, "profile_label"), ""),
        (("attempts", 0, "retry_task_keys"), [False]),
        (("attempts", 0, "duplicate_risk_task_keys"), ["task-0"]),
        (("attempts", 0, "target_digest"), 7),
        (("attempts", 0, "acceptance", "job_id"), 42.0),
        (("attempts", 0, "acceptance", "status"), "NOT_SUBMITTED"),
    ],
)
def test_campaign_rejects_malformed_bounded_state_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str | int, ...],
    value: object,
) -> None:
    campaign_path = tmp_path / "campaign"
    campaign = Campaign.open(campaign_path, tasks(1))
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, b"42\n", b""),
    )
    campaign.submit(planning(campaign))
    state_path = campaign_path / "campaign.json"
    state = json.loads(state_path.read_text())
    owner: object = state
    for component in path[:-1]:
        owner = owner[component]  # type: ignore[index]
    owner[path[-1]] = value  # type: ignore[index]
    state_path.write_text(json.dumps(state))

    with pytest.raises(TaskConflict):
        Campaign.load(campaign_path)


def test_campaign_rejects_sealed_phase_without_revision_change(tmp_path: Path) -> None:
    campaign_path = tmp_path / "campaign"
    Campaign.open(campaign_path, tasks(1))
    state_path = campaign_path / "campaign.json"
    state = json.loads(state_path.read_text())
    state["phase"] = "SEALED"
    state_path.write_text(json.dumps(state))

    with pytest.raises(TaskConflict, match="phase"):
        Campaign.load(campaign_path)


def test_campaign_rejects_revision_behind_attempt_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_path = tmp_path / "campaign"
    campaign = Campaign.open(campaign_path, tasks(1))
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, b"42\n", b""),
    )
    campaign.submit(planning(campaign))
    state_path = campaign_path / "campaign.json"
    state = json.loads(state_path.read_text())
    state["revision"] = 1
    state_path.write_text(json.dumps(state))

    with pytest.raises(TaskConflict, match="revision"):
        Campaign.load(campaign_path)


@pytest.mark.parametrize(
    ("revisions", "repeat_plan_digest"),
    [((-1, 2), False), ((0, 1), False), ((0, 0), False), ((0, 2), True)],
)
def test_campaign_rejects_impossible_attempt_revision_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    revisions: tuple[int, int],
    repeat_plan_digest: bool,
) -> None:
    campaign_path = tmp_path / "campaign"
    campaign = Campaign.open(campaign_path, tasks(1))
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, b"42\n", b""),
    )
    campaign.submit(planning(campaign))
    campaign.submit(planning(campaign, retry={"task-0"}))
    state_path = campaign_path / "campaign.json"
    state = json.loads(state_path.read_text())
    for attempt, revision in zip(state["attempts"], revisions, strict=True):
        attempt["campaign_revision"] = revision
    if repeat_plan_digest:
        state["attempts"][1]["plan_digest"] = state["attempts"][0]["plan_digest"]
    state_path.write_text(json.dumps(state))

    with pytest.raises(TaskConflict):
        Campaign.load(campaign_path)


@pytest.mark.parametrize("with_attempt", [False, True])
def test_campaign_rejects_unexplained_revision_inflation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_attempt: bool,
) -> None:
    campaign_path = tmp_path / "campaign"
    campaign = Campaign.open(campaign_path, tasks(1))
    if with_attempt:
        monkeypatch.setattr(
            _slurm,
            "_run_ssh",
            lambda *_args, **_kwargs: _slurm.Result(0, b"42\n", b""),
        )
        campaign.submit(planning(campaign))
    state_path = campaign_path / "campaign.json"
    state = json.loads(state_path.read_text())
    state["revision"] = 102 if with_attempt else 100
    if with_attempt:
        state["attempts"][0]["campaign_revision"] = 100
    state_path.write_text(json.dumps(state))

    with pytest.raises(TaskConflict, match="revision"):
        Campaign.load(campaign_path)


def test_campaign_rejects_lineage_without_attempt_history(tmp_path: Path) -> None:
    campaign_path = tmp_path / "campaign"
    Campaign.open(campaign_path, tasks(1))
    state_path = campaign_path / "campaign.json"
    state = json.loads(state_path.read_text())
    state["lineage"] = _campaign._lineage(target(), resources())
    state_path.write_text(json.dumps(state))

    with pytest.raises(TaskConflict, match="lineage"):
        Campaign.load(campaign_path)


@pytest.mark.parametrize("field", ["cpus", "memory_mib", "gpus"])
@pytest.mark.parametrize("value", [True, 1.0])
def test_campaign_rejects_noninteger_allocation_totals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: bool | float,
) -> None:
    path = tmp_path / "campaign"
    campaign = Campaign.open(path, tasks(1))
    request = resources(cpus_per_task=1, memory_mib_per_task=1, gpus_per_task=1)
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(1, b"", b"lost reply"),
    )
    with pytest.raises(AmbiguousSubmission):
        campaign.submit(planning(campaign, resource_value=request))

    state_path = path / "campaign.json"
    state = json.loads(state_path.read_text())
    state["attempts"][0]["allocation"][field] = value
    state_path.write_text(json.dumps(state))

    with pytest.raises(TaskConflict, match="provenance"):
        Campaign.open(path, tasks(1))


def test_campaign_rejects_reordered_attempt_task_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "campaign"
    campaign = Campaign.open(path, tasks(2))
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, b"42\n", b""),
    )
    campaign.submit(planning(campaign))
    state_path = path / "campaign.json"
    state = json.loads(state_path.read_text())
    state["attempts"][0]["task_keys"].reverse()
    state_path.write_text(json.dumps(state))

    with pytest.raises(TaskConflict, match="task keys"):
        Campaign.load(path)


@pytest.mark.parametrize("retry_task_keys", [[], ["task-0"]])
def test_campaign_rejects_erased_retry_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retry_task_keys: list[str],
) -> None:
    path = tmp_path / "campaign"
    campaign = Campaign.open(path, tasks(2))
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, b"42\n", b""),
    )
    campaign.submit(planning(campaign))
    campaign.submit(planning(campaign, retry={"task-0", "task-1"}))
    state_path = path / "campaign.json"
    state = json.loads(state_path.read_text())
    state["attempts"][1]["retry_task_keys"] = retry_task_keys
    state_path.write_text(json.dumps(state))

    with pytest.raises(TaskConflict, match="retry keys"):
        Campaign.load(path)


@pytest.mark.parametrize("corruption", ["empty", "noncanonical", "reversed"])
def test_campaign_rejects_invalid_reconciliation_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    path = tmp_path / "campaign"
    campaign = Campaign.open(path, tasks(1))
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(1, b"", b"lost reply"),
    )
    with pytest.raises(AmbiguousSubmission):
        campaign.submit(planning(campaign))
    state_path = path / "campaign.json"
    state = json.loads(state_path.read_text())
    attempt = state["attempts"][0]
    if corruption == "empty":
        attempt["window_start"] = ""
    elif corruption == "noncanonical":
        attempt["window_start"] = "2026-8-13T00:00:00"
    else:
        attempt["window_start"], attempt["window_end"] = (
            attempt["window_end"],
            attempt["window_start"],
        )
    state_path.write_text(json.dumps(state))

    with pytest.raises(TaskConflict, match="window"):
        Campaign.load(path)


def test_campaign_rejects_allocation_after_unresolved_outcome_in_same_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "campaign"
    campaign = Campaign.open(path, tasks(2))
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, b"42\n", b""),
    )
    campaign.submit(planning(campaign, target(max_tasks_per_allocation=1)))
    state_path = path / "campaign.json"
    state = json.loads(state_path.read_text())
    state["attempts"][0]["acceptance"] = {"status": "UNRESOLVED"}
    state_path.write_text(json.dumps(state))

    with pytest.raises(TaskConflict, match="plan outcome is terminal"):
        Campaign.load(path)


def test_campaign_rejects_attempt_after_not_submitted_plan_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "campaign"
    campaign = Campaign.open(path, tasks(2))
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, b"42\n", b""),
    )
    campaign.submit(planning(campaign, target(max_tasks_per_allocation=1)))
    state_path = path / "campaign.json"
    state = json.loads(state_path.read_text())
    state["attempts"][0]["acceptance"] = {"status": "NOT_SUBMITTED"}
    state_path.write_text(json.dumps(state))

    with pytest.raises(TaskConflict, match="terminal"):
        Campaign.load(path)


@pytest.mark.parametrize("corruption", ["reversed", "overlap"])
def test_campaign_rejects_changed_plan_allocation_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    path = tmp_path / "campaign"
    campaign = Campaign.open(path, tasks(2))
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, b"42\n", b""),
    )
    campaign.submit(planning(campaign, target(max_tasks_per_allocation=1)))
    state_path = path / "campaign.json"
    state = json.loads(state_path.read_text())
    if corruption == "reversed":
        state["attempts"].reverse()
    else:
        state["attempts"][1]["task_keys"] = ["task-0"]
        state["attempts"][1]["retry_task_keys"] = ["task-0"]
    state_path.write_text(json.dumps(state))

    with pytest.raises(TaskConflict, match="allocation sequence"):
        Campaign.load(path)


def test_intent_file_and_directory_are_synced_before_ssh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    plan = planning(campaign)
    campaign_entry = (tmp_path / "campaign").stat()
    events: list[str] = []
    real_fsync = _campaign.os.fsync
    real_replace = _campaign.os.replace

    def track_fsync(descriptor: int) -> None:
        entry = os.fstat(descriptor)
        if stat.S_ISREG(entry.st_mode):
            events.append("file-fsync")
        elif (entry.st_dev, entry.st_ino) == (campaign_entry.st_dev, campaign_entry.st_ino):
            events.append("directory-fsync")
        real_fsync(descriptor)

    def track_replace(*args: object, **kwargs: object) -> None:
        events.append("replace")
        real_replace(*args, **kwargs)  # type: ignore[arg-type]

    def accepted(*args: object, **kwargs: object) -> _slurm.Result:
        events.append("ssh")
        assert events.index("file-fsync") < events.index("replace")
        assert events.index("replace") < events.index("directory-fsync")
        assert events.index("directory-fsync") < events.index("ssh")
        return _slurm.Result(0, b"42\n", b"")

    monkeypatch.setattr(_campaign.os, "fsync", track_fsync)
    monkeypatch.setattr(_campaign.os, "replace", track_replace)
    monkeypatch.setattr(_slurm, "_run_ssh", accepted)
    campaign.submit(plan)


def test_accepted_without_receipt_is_ambiguous_and_halts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(5))
    plan = planning(campaign, replace(target(), max_tasks_per_allocation=2))
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
    assert (
        sum(
            attempt.acceptance is _campaign.AcceptanceState.UNRESOLVED
            for attempt in campaign.inspect(scheduler=False).attempts
        )
        == 1
    )


def test_manual_resolution_and_explicit_retry_preserve_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    plan = planning(campaign)
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *args, **kwargs: _slurm.Result(0, b"not-a-job\n", b""),
    )
    with pytest.raises(AmbiguousSubmission):
        campaign.submit(plan)

    allocation_id = campaign.inspect(scheduler=False).attempts[-1].allocation_id
    campaign.resolve(allocation_id, job_id=None)
    retry_plan = planning(campaign)
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *args, **kwargs: _slurm.Result(0, b"101\n", b""),
    )
    campaign.submit(retry_plan)
    second_retry = planning(campaign, retry={"task-0"})
    campaign.submit(second_retry)
    assert [
        attempt.receipt.job_id
        for attempt in campaign.inspect(scheduler=False).attempts
        if attempt.receipt is not None
    ] == [101, 101]


def test_attempt_history_retains_resolution_reconciliation_retry_and_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "campaign"
    campaign = Campaign.open(path, tasks(1))
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(1, b"", b"lost reply"),
    )
    with pytest.raises(AmbiguousSubmission):
        campaign.submit(planning(campaign))
    first = campaign.inspect(scheduler=False).attempts[-1].allocation_id
    campaign.resolve(first, job_id=None)

    with pytest.raises(AmbiguousSubmission):
        campaign.submit(planning(campaign))
    second = campaign.inspect(scheduler=False).attempts[-1].allocation_id
    monkeypatch.setattr(
        _slurm,
        "query_identity",
        lambda *_args, **_kwargs: _slurm.IdentityMatch(909, "alpha"),
    )
    assert campaign.reconcile(second).job_id == 909

    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, b"101\n", b""),
    )
    campaign.submit(planning(campaign, retry={"task-0"}))
    state = json.loads((path / "campaign.json").read_text())

    assert [attempt["campaign_revision"] for attempt in state["attempts"]] == [0, 2, 4]
    assert [attempt["retry_task_keys"] for attempt in state["attempts"]] == [
        [],
        [],
        ["task-0"],
    ]
    assert [attempt["acceptance"]["status"] for attempt in state["attempts"]] == [
        "NOT_SUBMITTED",
        "ACCEPTED",
        "ACCEPTED",
    ]
    assert [
        attempt.receipt.job_id
        for attempt in campaign.inspect(scheduler=False).attempts
        if attempt.receipt is not None
    ] == [909, 101]
    assert all(
        attempt["target_digest"] == state["lineage"]["target_digest"]
        and attempt["resource_digest"] == state["lineage"]["resource_digest"]
        for attempt in state["attempts"]
    )


def test_stale_foreign_and_tampered_plans_fail(tmp_path: Path) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    plan = planning(campaign)
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
    plan = planning(campaign, limited)
    calls = 0

    def accepted(*args: object, **kwargs: object) -> _slurm.Result:
        nonlocal calls
        calls += 1
        return _slurm.Result(0, f"{100 + calls}\n".encode(), b"")

    monkeypatch.setattr(_slurm, "_run_ssh", accepted)
    receipts = campaign.submit(plan)

    assert len(receipts) == 2
    assert calls == 2
    assert [allocation.task_keys for allocation in planning(campaign, limited).allocations] == [
        ("task-4",)
    ]
    with pytest.raises(PlanError, match="stale"):
        campaign.submit(plan)


def test_validate_deduplicates_shapes_and_never_mutates_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(5))
    plan = planning(campaign, target(max_tasks_per_allocation=2))
    before = (tmp_path / "campaign" / "campaign.json").read_bytes()
    calls: list[tuple[str, ...]] = []

    def validated(target_value: SlurmTarget, argv: tuple[str, ...], script: bytes) -> _slurm.Result:
        calls.append(argv)
        return _slurm.Result(0, b"Job 1 to start at 2030-01-01\n", b"")

    monkeypatch.setattr(_slurm, "_run_ssh", validated)
    results = campaign.validate(plan)

    assert len(results) == 2  # group shapes 2 and 1
    assert all(isinstance(result, ValidationResult) for result in results)
    assert results[0].task_count == 2
    assert results[0].cpus == 64
    assert all(argv[-1] == "--test-only" for argv in calls)
    assert before == (tmp_path / "campaign" / "campaign.json").read_bytes()


def test_validate_rejects_foreign_plan_before_contacting_slurm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = planning(Campaign.open(tmp_path / "source", tasks(1)))
    foreign = Campaign.open(tmp_path / "foreign", tasks(1))
    called = False

    def contacted(*args: object, **kwargs: object) -> _slurm.Result:
        nonlocal called
        called = True
        return _slurm.Result(0, b"valid\n", b"")

    monkeypatch.setattr(_slurm, "_run_ssh", contacted)

    with pytest.raises(PlanError, match="another campaign") as rejected:
        foreign.validate(plan)
    assert "validate" in str(rejected.value)
    assert "submit" not in str(rejected.value)
    assert "submission" not in str(rejected.value)
    assert called is False


def test_reconcile_adopts_only_private_query_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "campaign"
    campaign = Campaign.open(path, tasks(1))
    plan = planning(campaign)
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *args, **kwargs: _slurm.Result(1, b"", b"lost reply"),
    )
    with pytest.raises(AmbiguousSubmission):
        campaign.submit(plan)
    allocation_id = campaign.inspect(scheduler=False).attempts[-1].allocation_id
    Campaign.open(path, tasks(2))
    queried_targets: list[SlurmTarget] = []

    def query(target_value: SlurmTarget, **kwargs: object) -> _slurm.IdentityMatch:
        queried_targets.append(target_value)
        return _slurm.IdentityMatch(909, "alpha")

    monkeypatch.setattr(_slurm, "query_identity", query)

    receipt = campaign.reconcile(allocation_id)

    assert (receipt.job_id, receipt.cluster) == (909, "alpha")
    assert queried_targets == [target()]
    evidence = campaign.inspect(scheduler=False)
    assert all(
        attempt.acceptance is not _campaign.AcceptanceState.UNRESOLVED
        for attempt in evidence.attempts
    )
    assert [allocation.task_keys for allocation in planning(campaign).allocations] == [("task-1",)]


def test_state_size_boundary_is_symmetric_and_overflow_does_not_mutate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    state_path = tmp_path / "campaign" / "campaign.json"
    with campaign._locked_state() as state:
        assert state is not None
        exact = len(_campaign._canonical(state) + b"\n")
        monkeypatch.setattr(_campaign, "_MAX_STATE_BYTES", exact)
        campaign._write_state(state)
        accepted = state_path.read_bytes()
        monkeypatch.setattr(_campaign, "_MAX_STATE_BYTES", exact - 1)
        with pytest.raises(TaskConflict, match="maximum"):
            campaign._write_state(state)
        assert state_path.read_bytes() == accepted
        assert list((tmp_path / "campaign").glob(".campaign-*.tmp")) == []


def test_campaign_rejects_invalid_json_state(tmp_path: Path) -> None:
    path = tmp_path / "campaign"
    Campaign.open(path, tasks(1))
    (path / "campaign.json").write_bytes(b'{"truncated":')

    with pytest.raises(TaskConflict, match="invalid JSON"):
        Campaign.open(path, tasks(1))


def test_campaign_rejects_truncated_state_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    real_read = _campaign.os.read
    first = True

    def truncated_read(descriptor: int, count: int) -> bytes:
        nonlocal first
        if first:
            first = False
            return real_read(descriptor, count)[:-1]
        return b""

    monkeypatch.setattr(_campaign.os, "read", truncated_read)
    with pytest.raises(TaskConflict, match="changed while reading"):
        _ = campaign.tasks


def test_campaign_rejects_oversized_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "campaign"
    campaign = Campaign.open(path, tasks(1))
    state_path = path / "campaign.json"
    monkeypatch.setattr(_campaign, "_MAX_STATE_BYTES", state_path.stat().st_size - 1)
    with pytest.raises(TaskConflict, match="too large"):
        _ = campaign.tasks


def test_campaign_rejects_permissive_state_file(tmp_path: Path) -> None:
    path = tmp_path / "campaign"
    campaign = Campaign.open(path, tasks(1))
    state_path = path / "campaign.json"
    state_path.chmod(0o640)
    with pytest.raises(TaskConflict, match="owner-only"):
        _ = campaign.tasks


@pytest.mark.parametrize("failure", ["write", "fsync"])
def test_precommit_state_failures_remove_owned_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    state_path = tmp_path / "campaign" / "campaign.json"
    before = state_path.read_bytes()
    real_write = _campaign.os.write
    real_fsync = _campaign.os.fsync

    def fail_write(descriptor: int, data: object) -> int:
        raise OSError("injected state write failure")

    def fail_stage_sync(descriptor: int) -> None:
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("injected state fsync failure")
        real_fsync(descriptor)

    with campaign._locked_state() as state:
        assert state is not None
        if failure == "write":
            monkeypatch.setattr(_campaign.os, "write", fail_write)
        else:
            monkeypatch.setattr(_campaign.os, "fsync", fail_stage_sync)
        with pytest.raises(OSError, match="injected state"):
            campaign._write_state(state)
    monkeypatch.setattr(_campaign.os, "write", real_write)
    assert state_path.read_bytes() == before
    assert list((tmp_path / "campaign").glob(".campaign-*.tmp")) == []


def test_campaign_creation_syncs_parent_before_state_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    parent_entry = tmp_path.stat()
    real_mkdir = _campaign.os.mkdir
    real_fsync = _campaign.os.fsync
    real_replace = _campaign.os.replace

    def track_mkdir(*args: object, **kwargs: object) -> None:
        events.append("mkdir")
        real_mkdir(*args, **kwargs)  # type: ignore[arg-type]

    def track_fsync(descriptor: int) -> None:
        entry = os.fstat(descriptor)
        if (entry.st_dev, entry.st_ino) == (parent_entry.st_dev, parent_entry.st_ino):
            events.append("parent-fsync")
        real_fsync(descriptor)

    def track_replace(*args: object, **kwargs: object) -> None:
        events.append("state-replace")
        real_replace(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_campaign.os, "mkdir", track_mkdir)
    monkeypatch.setattr(_campaign.os, "fsync", track_fsync)
    monkeypatch.setattr(_campaign.os, "replace", track_replace)
    Campaign.open(tmp_path / "campaign", tasks(1))
    assert events.index("mkdir") < events.index("parent-fsync") < events.index("state-replace")


def test_campaign_parent_sync_failure_reports_and_removes_unsynced_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_entry = tmp_path.stat()
    real_fsync = _campaign.os.fsync

    def fail_parent(descriptor: int) -> None:
        entry = os.fstat(descriptor)
        if (entry.st_dev, entry.st_ino) == (parent_entry.st_dev, parent_entry.st_ino):
            raise OSError("injected parent fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(_campaign.os, "fsync", fail_parent)
    with pytest.raises(OSError, match="parent fsync failure"):
        Campaign.open(tmp_path / "campaign", tasks(1))
    assert not (tmp_path / "campaign").exists()


def test_concurrent_observer_proves_parent_durability_while_creator_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_entry = tmp_path.stat()
    creator_at_sync = threading.Event()
    release_creator = threading.Event()
    observer_synced = threading.Event()
    real_fsync = _campaign.os.fsync
    results: dict[str, Campaign] = {}
    failures: list[BaseException] = []

    def controlled_fsync(descriptor: int) -> None:
        entry = os.fstat(descriptor)
        is_parent = (entry.st_dev, entry.st_ino) == (parent_entry.st_dev, parent_entry.st_ino)
        if is_parent and threading.current_thread().name == "creator":
            creator_at_sync.set()
            assert release_creator.wait(timeout=5)
        if is_parent and threading.current_thread().name == "observer":
            observer_synced.set()
        real_fsync(descriptor)

    def open_campaign(name: str) -> None:
        try:
            results[name] = Campaign.open(tmp_path / "campaign", tasks(1))
        except BaseException as error:
            failures.append(error)

    monkeypatch.setattr(_campaign.os, "fsync", controlled_fsync)
    creator = threading.Thread(target=open_campaign, args=("creator",), name="creator")
    creator.start()
    assert creator_at_sync.wait(timeout=5)
    observer = threading.Thread(target=open_campaign, args=("observer",), name="observer")
    observer.start()
    observer.join(timeout=5)

    assert not observer.is_alive()
    assert observer_synced.is_set()
    assert "observer" in results
    assert "creator" not in results
    assert failures == []

    release_creator.set()
    creator.join(timeout=5)
    assert not creator.is_alive()
    assert set(results) == {"creator", "observer"}
    assert results["creator"].tasks == results["observer"].tasks
    assert list((tmp_path / "campaign").glob(".campaign-*.tmp")) == []


def test_concurrent_openers_cannot_return_when_all_parent_syncs_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_entry = tmp_path.stat()
    creator_at_sync = threading.Event()
    release_creator = threading.Event()
    returned: list[str] = []
    failures: list[tuple[str, BaseException]] = []
    creator_parent_calls = 0
    real_fsync = _campaign.os.fsync

    def fail_parent_sync(descriptor: int) -> None:
        nonlocal creator_parent_calls
        entry = os.fstat(descriptor)
        is_parent = (entry.st_dev, entry.st_ino) == (parent_entry.st_dev, parent_entry.st_ino)
        if not is_parent:
            real_fsync(descriptor)
            return
        if threading.current_thread().name == "creator":
            creator_parent_calls += 1
            if creator_parent_calls == 1:
                creator_at_sync.set()
                assert release_creator.wait(timeout=5)
        raise OSError("injected concurrent parent fsync failure")

    def open_campaign(name: str) -> None:
        try:
            Campaign.open(tmp_path / "campaign", tasks(1))
            returned.append(name)
        except BaseException as error:
            failures.append((name, error))

    monkeypatch.setattr(_campaign.os, "fsync", fail_parent_sync)
    creator = threading.Thread(target=open_campaign, args=("creator",), name="creator")
    creator.start()
    assert creator_at_sync.wait(timeout=5)
    observer = threading.Thread(target=open_campaign, args=("observer",), name="observer")
    observer.start()
    observer.join(timeout=5)
    assert not observer.is_alive()
    assert returned == []

    release_creator.set()
    creator.join(timeout=5)
    assert not creator.is_alive()
    assert returned == []
    assert {name for name, _ in failures} == {"creator", "observer"}
    assert all("concurrent parent fsync failure" in str(error) for _, error in failures)
    assert not (tmp_path / "campaign").exists()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), True),
        (("resources", "cpus_per_task"), True),
        (("target", "max_tasks_per_allocation"), True),
        (("allocations", 0, "cpus"), True),
        (("view", "scheduler_observed"), 1),
        (("tasks_per_allocation",), True),
    ],
)
def test_plan_document_rejects_invalid_inputs_and_derived_tampering(
    tmp_path: Path, path: tuple[str | int, ...], value: object
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    document = copy.deepcopy(_campaign.plan_document(planning(campaign)))
    owner: object = document
    for component in path[:-1]:
        owner = owner[component]  # type: ignore[index]
    owner[path[-1]] = value  # type: ignore[index]
    with pytest.raises(PlanError):
        _campaign.restore_plan(campaign, document)
