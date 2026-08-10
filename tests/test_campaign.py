from __future__ import annotations

import base64
import copy
import json
import os
import stat
import threading
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


@pytest.mark.parametrize(
    "gres",
    ["scratch", "gpu:2", "gpu:a100:2", "gpu:a100/bad", "gpu:a100\n"],
)
def test_gpu_gres_accepts_only_count_free_gpu_family(gres: str) -> None:
    with pytest.raises(ConfigurationError):
        target(gpu_gres=gres)


def test_target_toml_rejects_non_gpu_gres_family(tmp_path: Path) -> None:
    path = tmp_path / "target.toml"
    path.write_text(
        'host = "login.example.edu"\nslurm_bin = "/opt/slurm/bin"\n'
        'apptainer = "/usr/bin/apptainer"\nimage = "/images/work.sif"\n'
        'work_root = "/work"\nlog_root = "/logs"\npartitions = ["gpu"]\n'
        'gpu_gres = "scratch"\nmax_tasks_per_allocation = 1\n'
        "max_cpus_per_allocation = 8\nmax_memory_mib_per_allocation = 8192\n"
        'max_gpus_per_allocation = 1\nmax_time_limit = "01:00:00"\n'
        "max_allocations_per_submit = 1\nmax_script_bytes = 1048576\n"
    )
    with pytest.raises(ConfigurationError):
        SlurmTarget.from_toml(path)


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
    stale = campaign.plan(target(), resources())
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
    campaign.submit(campaign.plan(target(), resources()))
    grown = Campaign.open(path, tasks(4))
    suffix_plan = grown.plan(target(), resources())

    assert [allocation.task_keys for allocation in suffix_plan.allocations] == [
        ("task-2", "task-3")
    ]
    grown.submit(suffix_plan)
    assert [receipt.task_keys for receipt in grown.status().receipts] == [
        ("task-0", "task-1"),
        ("task-2", "task-3"),
    ]
    assert grown.status().pending_task_keys == ()
    retry = grown.plan(target(), resources(), retry={"task-0"})
    assert [allocation.task_keys for allocation in retry.allocations] == [("task-0",)]


def test_growth_preserves_ambiguous_intent_and_resource_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "campaign"
    campaign = Campaign.open(path, tasks(1))

    def lose_reply(*_args: object, **_kwargs: object) -> _slurm.Result:
        raise OSError("lost")

    monkeypatch.setattr(_slurm, "_run_ssh", lose_reply)
    with pytest.raises(AmbiguousSubmission):
        campaign.submit(campaign.plan(target(), resources()))
    ambiguous = campaign.status().ambiguous_allocation_ids

    grown = Campaign.open(path, tasks(2))

    assert grown.status().ambiguous_allocation_ids == ambiguous
    with pytest.raises(AmbiguousSubmission):
        grown.plan(target(), resources())
    grown.resolve(ambiguous[0], job_id=None)
    with pytest.raises(PlanError, match="resource semantics"):
        grown.plan(target(), resources(cpus_per_task=16))
    assert tuple(
        key
        for allocation in grown.plan(target(), resources()).allocations
        for key in allocation.task_keys
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
    assert recovered.status().pending_task_keys == ("task-0", "task-1")
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
        {"max_time_limit": "2-23:58:59"},
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


def test_effective_slurm_time_is_rounded_once_and_requested_time_is_preserved(
    tmp_path: Path,
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(4))
    plan = campaign.plan(
        target(max_time_limit="00:00:01"),
        resources(time_limit="00:00:30"),
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

    seconds = Campaign.open(tmp_path / "seconds", tasks(1)).plan(
        target(), resources(time_limit="01:02:03")
    )
    assert seconds.allocations[0].time_limit == "01:03:00"
    assert "--time=01:03:00" in seconds._allocations[0].argv

    with pytest.raises(PlanError, match="time_limit"):
        Campaign.open(tmp_path / "too-long", tasks(1)).plan(
            target(max_time_limit="00:01:00"), resources(time_limit="00:01:01")
        )


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
        assert state["lineage"]["target"] == _campaign._target_dict(target())
        assert state["lineage"]["resources"] == _campaign._resource_dict(resources())
        assert state["intents"][0]["allocation"] == {
            "cpus": 32,
            "memory_mib": 65536,
            "gpus": 1,
            "time_limit": "3-00:00:00",
        }
        assert state["intents"][0]["sbatch_argv"] == list(argv)
        assert script == plan._allocations[0].script
        return _slurm.Result(0, b"4242;alpha\n", b"")

    monkeypatch.setattr(_slurm, "_run_ssh", accepted)
    receipts = campaign.submit(plan)

    assert [(receipt.job_id, receipt.cluster) for receipt in receipts] == [(4242, "alpha")]
    assert campaign.status().pending_task_keys == ()
    state_path = tmp_path / "campaign" / "campaign.json"
    assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
    assert list((tmp_path / "campaign").glob(".campaign-*.tmp")) == []


def test_intent_file_and_directory_are_synced_before_ssh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    plan = campaign.plan(target(), resources())
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
    assert results["creator"].status() == results["observer"].status()
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
        (("state_revision",), 0.0),
        (("resources", "cpus_per_task"), True),
        (("target", "max_tasks_per_allocation"), True),
        (("allocations", 0, "cpus"), True),
        (("allocations", 0, "sbatch_argv"), [True]),
        (("allocations", 0, "sbatch_argv", 0), "changed"),
        (("allocations", 0, "argv_digest"), "0" * 64),
        (("allocations",), {}),
        (("completed",), 7),
    ],
)
def test_plan_document_rejects_changed_container_and_numeric_types(
    tmp_path: Path, path: tuple[str | int, ...], value: object
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    document = copy.deepcopy(_campaign.plan_document(campaign.plan(target(), resources())))
    owner: object = document
    for component in path[:-1]:
        owner = owner[component]  # type: ignore[index]
    owner[path[-1]] = value  # type: ignore[index]
    with pytest.raises(PlanError):
        _campaign.restore_plan(campaign, document)
