"""Planning through the public API: capacity, packing, eligibility, and saved plans."""

from __future__ import annotations

import json
import stat
from dataclasses import replace
from datetime import timedelta
from typing import Any

import pytest
from campaign_world import Probe, World, cpu_profile, jobs
from support.builders import profile, resources, target

from servatus.campaign import Completed, Hold, Profile, Retry, Task
from servatus.errors import (
    ConfigurationError,
    DestinationExists,
    PlanRefused,
    StalePlan,
    SubmissionInterrupted,
    Unavailable,
)


def keys(*indices: int) -> tuple[str, ...]:
    return tuple(f"task-{index}" for index in indices)


# --- Capacity and packing ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"max_cpus_per_allocation": 1}, "max_cpus_per_allocation"),
        ({"max_memory_mib_per_allocation": 1000}, "max_memory_mib_per_allocation"),
        ({"max_time_limit": "00:30:00"}, "max_time_limit"),
    ],
)
def test_every_target_ceiling_can_refuse_one_task_before_any_observation(
    world: World, changes: dict[str, Any], message: str
) -> None:
    probe = Probe()
    campaign = world.create(jobs(1), probe=probe)
    with pytest.raises(ConfigurationError, match=message):
        campaign.plan(cpu_profile(**changes))
    with pytest.raises(ConfigurationError, match="exceeds the feasible capacity 4"):
        campaign.plan(cpu_profile(), tasks_per_allocation=5)
    assert probe.calls == [] and world.scheduler_calls() == 0


def test_gpu_work_needs_a_gpu_resource(world: World) -> None:
    campaign = world.create(jobs(1))
    gpu = Profile(
        "gpu", cpu_profile().target, resources(cpus=1, memory_mib=1, gpus=1, time_limit="01:00:00")
    )
    with pytest.raises(ConfigurationError, match="gpu_gres"):
        campaign.plan(gpu)


@pytest.mark.parametrize("count", range(1, 10))
def test_balanced_allocations_preserve_order_and_exact_requests(world: World, count: int) -> None:
    campaign = world.create(jobs(count))
    plan = campaign.plan(cpu_profile())
    sizes = [len(item.task_keys) for item in plan.allocations]
    assert sizes == sorted(sizes, reverse=True) and max(sizes) - min(sizes) <= 1
    assert tuple(key for item in plan.allocations for key in item.task_keys) == keys(*range(count))
    for item in plan.allocations:
        size = len(item.task_keys)
        assert (item.cpus, item.memory_mib, item.gpus) == (2 * size, 1024 * size, 0)
        assert item.time_limit == timedelta(hours=1)
        assert f"--ntasks={size}" in item.argv and f"--mem={1024 * size}M" in item.argv
        assert item.script.count(b"--cpus-per-task=2") == size


def test_gpu_allocations_request_gres_per_allocation_and_per_step(world: World) -> None:
    campaign = world.create(tuple(Task(f"g-{index}", ("train",)) for index in range(4)))
    plan = campaign.plan(profile())
    (item,) = plan.allocations
    assert (item.cpus, item.memory_mib, item.gpus) == (128, 262144, 4)
    assert "--gres=gpu:a100:4" in item.argv and "--nodes=1" in item.argv
    assert item.script.count(b"--gres=gpu:a100:1") == 4


def test_batch_cap_defers_eligible_tasks_and_keeps_retry_intent(world: World) -> None:
    campaign = world.create(jobs(11))
    capped = cpu_profile(max_allocations_per_submit=2)
    plan = campaign.plan(capped, tasks_per_allocation=3)
    assert [len(item.task_keys) for item in plan.allocations] == [3, 3]
    assert plan.selected == keys(*range(6)) and plan.deferred == keys(*range(6, 11))
    assert any("deferred" in warning for warning in plan.warnings)
    campaign.submit(plan)
    assert campaign.plan(capped, tasks_per_allocation=3).selected == keys(*range(6, 11))

    for job in (1000, 1001):
        world.fake.finish(job, "FAILED", exit_code="1:0")
    single = cpu_profile(max_allocations_per_submit=1)
    retry = campaign.plan(single, retry=Retry.FAILED, tasks_per_allocation=2)
    assert retry.selected == keys(0, 1)
    assert retry.deferred == keys(*range(2, 11))
    assert retry.retry == keys(*range(6))  # deferred retries stay visible
    campaign.submit(retry)
    following = campaign.plan(single, retry=Retry.FAILED, tasks_per_allocation=2)
    assert following.selected == keys(2, 3)
    assert following.held["task-0"] is Hold.ACTIVE
    explicit = campaign.plan(single, retry=keys(*range(6)), tasks_per_allocation=2)
    assert explicit.selected == keys(2, 3) and explicit.held["task-1"] is Hold.ACTIVE


def test_unbounded_targets_plan_everything(world: World) -> None:
    campaign = world.create(jobs(9))
    plan = campaign.plan(cpu_profile(), tasks_per_allocation=1)
    assert len(plan.allocations) == 9 and plan.deferred == ()


# --- Eligibility -------------------------------------------------------------------------------


def test_fresh_only_plans_never_contact_historical_routes(world: World) -> None:
    campaign = world.create(jobs(1), appendable=True)
    campaign.submit(campaign.plan(cpu_profile(host="old.example.edu")))
    campaign.append(jobs(2)[1:])
    world.wire.down.add("old.example.edu")
    calls, routes = world.scheduler_calls(), len(world.wire.hosts)
    plan = campaign.plan(cpu_profile(host="new.example.edu"))
    assert plan.selected == ("task-1",) and dict(plan.held) == {"task-0": Hold.SUBMITTED}
    assert world.scheduler_calls() == calls and len(world.wire.hosts) == routes
    with pytest.raises(Unavailable, match="Connection refused"):
        campaign.plan(cpu_profile(host="new.example.edu"), retry=["task-0"])


def test_every_hold_reason_is_reported(world: World) -> None:
    probe = Probe()
    campaign = world.create(jobs(6), probe=probe)
    campaign.submit(campaign.plan(cpu_profile(), tasks_per_allocation=1))
    world.fake.start(1000)  # task-0 active
    world.fake.finish(1001)  # task-1 succeeded, result valid
    world.fake.finish(1002, "FAILED", exit_code="1:0")  # task-2 failed
    world.fake.forget(1003)  # task-3 unknown
    world.fake.finish(1004)  # task-4 succeeded, result missing
    world.fake.finish(1005, "FAILED", exit_code="1:0")  # task-5 failed, then unresolved
    probe.valid.add("task-1")
    world.wire.on("sbatch", lambda _argv: (_ for _ in ()).throw(Unavailable("reset")))
    with pytest.raises(SubmissionInterrupted, match="unresolved"):
        campaign.submit(campaign.plan(cpu_profile(), retry=["task-5"]))

    bulk = campaign.plan(cpu_profile(), retry=Retry.FAILED)
    assert bulk.selected == ("task-2",)
    assert dict(bulk.held) == {
        "task-0": Hold.ACTIVE,
        "task-1": Hold.VALID,
        "task-3": Hold.UNOBSERVABLE,
        "task-4": Hold.SUBMITTED,
        "task-5": Hold.UNRESOLVED,
    }
    only = campaign.plan(cpu_profile(), retry=["task-0", "task-2"], only=["task-0", "task-2"])
    assert only.selected == ("task-2",) and only.held["task-0"] is Hold.ACTIVE
    assert only.held["task-4"] is Hold.NOT_REQUESTED
    incomplete = campaign.plan(cpu_profile(), retry=Retry.INCOMPLETE)
    assert incomplete.selected == ("task-2", "task-4")


@pytest.mark.parametrize(
    ("retry", "acknowledged", "message"),
    [
        (["nope"], [], "unknown Task keys: 'nope'"),
        (["task-1"], [], "retry needs an earlier accepted attempt: 'task-1'"),
        (["task-0"], [], "needs a duplicate-risk acknowledgement: 'task-0'"),
        ([], ["task-0"], "requires an explicitly retried Task key: 'task-0'"),
    ],
)
def test_explicit_retries_are_refused_with_every_reason(
    world: World, retry: list[str], acknowledged: list[str], message: str
) -> None:
    campaign = world.create(jobs(2), appendable=True)
    campaign.submit(campaign.plan(cpu_profile(), only=["task-0"]))
    world.fake.forget(1000)
    with pytest.raises(PlanRefused, match=message):
        campaign.plan(cpu_profile(), retry=retry, allow_duplicate_risk=acknowledged)
    plan = campaign.plan(cpu_profile(), retry=["task-0"], allow_duplicate_risk=["task-0"])
    assert plan.duplicate_risk == ("task-0",) and plan.warnings


def test_valid_results_are_never_retried(world: World) -> None:
    world.create(jobs(1))
    campaign = world.open(probe=Probe({"task-0"}))
    world.open().submit(world.open().plan(cpu_profile()))  # planned and submitted without a probe
    world.fake.finish(1000)
    with pytest.raises(PlanRefused, match="valid results cannot be retried: 'task-0'"):
        campaign.plan(cpu_profile(), retry=["task-0"])


def test_selector_shapes_and_probe_requirements_are_checked(world: World) -> None:
    campaign = world.create(jobs(1))
    with pytest.raises(ConfigurationError, match="retry must be a collection"):
        campaign.plan(cpu_profile(), retry="task-0")
    with pytest.raises(ConfigurationError, match="only must contain only strings"):
        campaign.plan(cpu_profile(), only=[1])  # pyright: ignore[reportArgumentType]
    with pytest.raises(ConfigurationError, match="Retry.INCOMPLETE needs a result probe"):
        campaign.plan(cpu_profile(), retry=Retry.INCOMPLETE)
    with pytest.raises(ConfigurationError, match="profile must be a Profile"):
        campaign.plan(cpu_profile().target)  # pyright: ignore[reportArgumentType]


# --- Rendering bounds --------------------------------------------------------------------------


def test_script_size_boundary_is_exact(world: World) -> None:
    campaign = world.create(jobs(1))
    size = len(campaign.plan(cpu_profile()).allocations[0].script)
    campaign.plan(cpu_profile(max_script_bytes=size))
    with pytest.raises(ConfigurationError, match="max_script_bytes"):
        campaign.plan(cpu_profile(max_script_bytes=size - 1))


def test_rendering_failures_are_refused_while_planning(world: World) -> None:
    campaign = world.create([Task("relative", ("python3", "train.py"))])
    with pytest.raises(ConfigurationError, match="absolute program path"):
        campaign.plan(cpu_profile())
    partitions = tuple(f"partition-{index:04d}" for index in range(300))
    with pytest.raises(ConfigurationError, match="command argument exceeds"):
        campaign.plan(replace(cpu_profile(), target=target(partitions=partitions)))
    assert campaign.status(scheduler=False).attempts == ()


# --- Saved plans -------------------------------------------------------------------------------


def test_saved_plans_are_compact_private_and_round_trip_without_observation(world: World) -> None:
    campaign = world.create(jobs(3), probe=Probe())
    plan = campaign.plan(cpu_profile())
    data = plan.to_json()
    document = json.loads(data)
    assert document["format"] == "servatus.plan/1" and document["digest"] == plan.digest
    assert document["selected"] == list(keys(0, 1, 2)) and document["probe_required"] is True
    assert all(secret not in data for secret in (b"secret-", b"train.py", b"SEED"))
    destination = world.root / "plan.json"
    plan.save(destination)
    assert destination.read_bytes() == data
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    with pytest.raises(DestinationExists, match="plan.json"):
        plan.save(destination)
    calls = world.scheduler_calls()
    assert world.open().load_plan(destination.read_bytes()) == plan
    assert world.scheduler_calls() == calls


def test_plans_of_the_same_state_have_distinct_identities(world: World) -> None:
    campaign = world.create(jobs(2))
    first, second = campaign.plan(cpu_profile()), campaign.plan(cpu_profile())
    assert first.selected == second.selected
    assert first.allocations[0].allocation_id != second.allocations[0].allocation_id
    assert first.digest != second.digest


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("digest", "a" * 64, ConfigurationError, "digest does not match"),
        ("digest", "short", ConfigurationError, "invalid plan document"),
        ("revision", 9, StalePlan, "changed after planning"),
        ("revision", True, ConfigurationError, "invalid plan document"),
        ("campaign_id", "b" * 32, StalePlan, "different campaign"),
        ("selected", ["task-0"], ConfigurationError, "partition the campaign roster"),
        ("probe_required", True, ConfigurationError, "digest does not match"),
        ("nonce", "c" * 32, ConfigurationError, "digest does not match"),
        ("retry", ["task-0"], ConfigurationError, "retry keys disagree"),
        ("format", "servatus.plan/0", ConfigurationError, "expected format"),
        ("surprise", 1, ConfigurationError, "unknown keys"),
    ],
)
def test_changed_plan_documents_are_refused(
    world: World, field: str, value: object, error: type[Exception], message: str
) -> None:
    campaign = world.create(jobs(2))
    document = json.loads(campaign.plan(cpu_profile()).to_json())
    document[field] = value
    with pytest.raises(error, match=message):
        campaign.load_plan(json.dumps(document).encode())


@pytest.mark.parametrize("data", [b"", b"[]", b"{", b'{"format": "servatus.plan/1"}'])
def test_malformed_plan_documents_are_refused(world: World, data: bytes) -> None:
    campaign = world.create(jobs(1))
    with pytest.raises(ConfigurationError, match="invalid plan document"):
        campaign.load_plan(data)


def test_a_saved_plan_goes_stale_when_the_campaign_changes(world: World) -> None:
    campaign = world.create(jobs(1), appendable=True)
    data = campaign.plan(cpu_profile()).to_json()
    campaign.append(jobs(2)[1:])
    with pytest.raises(StalePlan, match="revision 0 is now 1"):
        campaign.load_plan(data)


def test_validate_checks_each_distinct_shape_without_recording(world: World) -> None:
    campaign = world.create(jobs(5))
    plan = campaign.plan(cpu_profile())
    assert [len(item.task_keys) for item in plan.allocations] == [3, 2]
    checks = campaign.validate(plan)
    assert [(check.task_count, check.cpus, check.accepted) for check in checks] == [
        (3, 6, True),
        (2, 4, True),
    ]
    assert "to start at" in checks[0].scheduler_stderr
    tested = [call for call in world.fake.calls if "--test-only" in call]
    assert len(tested) == 2 and tested[0][:-1] == plan.allocations[0].argv
    rejection = "sbatch: error: Requested node configuration is not available"
    world.wire.on("sbatch", lambda _argv: Completed(1, b"", f"{rejection}\n".encode()))
    (rejected, _) = campaign.validate(plan)
    assert not rejected.accepted and rejected.scheduler_stderr == rejection
    assert campaign.status(scheduler=False).attempts == () and world.fake.jobs == ()
