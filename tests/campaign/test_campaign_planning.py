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

from servatus.campaign import Completed, Hold, Profile, Retry, Task, Totals
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
    assert plan.totals == Totals(cpu_hours=9216.0, gpu_hours=288.0, memory_gib_hours=18432.0)


def test_batch_cap_defers_eligible_tasks_and_keeps_retry_intent(world: World) -> None:
    campaign = world.create(jobs(11))
    capped = cpu_profile(max_allocations_per_submit=2)
    plan = campaign.plan(capped, tasks_per_allocation=3)
    assert [len(item.task_keys) for item in plan.allocations] == [3, 3]
    assert plan.decision.selected == keys(*range(6)) and plan.decision.deferred == keys(
        *range(6, 11)
    )
    assert any("deferred" in warning for warning in plan.warnings)
    campaign.submit(plan)
    assert campaign.plan(capped, tasks_per_allocation=3).decision.selected == keys(*range(6, 11))

    for job in (1000, 1001):
        world.fake.finish(job, exit_code=1)
        world.fake.purge(job)
    single = cpu_profile(max_allocations_per_submit=1)
    retry = campaign.plan(single, retry=Retry.FAILED, tasks_per_allocation=2)
    assert retry.decision.selected == keys(0, 1)
    assert retry.decision.deferred == keys(*range(2, 11))
    assert retry.retry == keys(*range(6))  # deferred retries stay visible
    campaign.submit(retry)
    following = campaign.plan(single, retry=Retry.FAILED, tasks_per_allocation=2)
    assert following.decision.selected == keys(2, 3)
    assert following.decision.held["task-0"] is Hold.ACTIVE
    explicit = campaign.plan(single, retry=keys(*range(2, 6)), tasks_per_allocation=2)
    assert explicit.decision.selected == keys(2, 3)
    with pytest.raises(PlanRefused, match=r"Slurm still lists earlier work.*'task-0', 'task-1'"):
        campaign.plan(single, retry=keys(*range(6)), tasks_per_allocation=2)


def test_unbounded_targets_plan_everything(world: World) -> None:
    campaign = world.create(jobs(9))
    plan = campaign.plan(cpu_profile(), tasks_per_allocation=1)
    assert len(plan.allocations) == 9 and plan.decision.deferred == ()


# --- Eligibility -------------------------------------------------------------------------------


def test_fresh_only_plans_never_contact_historical_routes(world: World) -> None:
    campaign = world.create(jobs(1), appendable=True)
    campaign.submit(campaign.plan(cpu_profile(host="old.example.edu")))
    campaign.append(jobs(2)[1:])
    world.wire.down.add("old.example.edu")
    calls, routes = world.scheduler_calls(), len(world.wire.hosts)
    plan = campaign.plan(cpu_profile(host="new.example.edu"))
    assert plan.decision.selected == ("task-1",) and dict(plan.decision.held) == {
        "task-0": Hold.SUBMITTED
    }
    assert world.scheduler_calls() == calls and len(world.wire.hosts) == routes
    with pytest.raises(Unavailable, match="Connection refused"):
        campaign.plan(cpu_profile(host="new.example.edu"), retry=["task-0"])


def test_plans_of_fresh_work_make_no_scheduler_calls(world: World) -> None:
    campaign = world.create(jobs(3), probe=Probe())
    plan = campaign.plan(cpu_profile(), only=["task-0"])
    assert world.scheduler_calls() == 0
    campaign.submit(plan)
    calls = world.scheduler_calls()
    for retry in ((), Retry.FAILED):
        following = campaign.plan(cpu_profile(), retry=retry, only=["task-1", "task-2"])
        assert following.decision.selected == ("task-1", "task-2")
    assert world.scheduler_calls() == calls
    campaign.plan(cpu_profile(), retry=Retry.FAILED)
    assert world.scheduler_calls() > calls  # bulk retry observes the accepted task-0


def test_every_hold_reason_is_reported(world: World) -> None:
    probe = Probe()
    campaign = world.create(jobs(6), probe=probe)
    campaign.submit(campaign.plan(cpu_profile(), tasks_per_allocation=1))
    world.fake.start(1000)  # task-0 running
    world.fake.finish(1001)  # task-1 succeeded, result valid
    world.fake.finish(1002, exit_code=1)  # task-2 failed
    world.fake.start(1003)
    world.fake.lose(1003)  # task-3 lost
    world.fake.finish(1004)  # task-4 succeeded, result missing
    world.fake.finish(1005, exit_code=1)  # task-5 failed, then unresolved
    for job in range(1001, 1006):
        world.fake.purge(job)
    probe.valid.add("task-1")
    world.fake.fail_next("sbatch")
    with pytest.raises(SubmissionInterrupted, match="unresolved"):
        campaign.submit(campaign.plan(cpu_profile(), retry=["task-5"]))

    bulk = campaign.plan(cpu_profile(), retry=Retry.FAILED)
    assert bulk.decision.selected == ("task-2",)
    assert dict(bulk.decision.held) == {
        "task-0": Hold.ACTIVE,
        "task-1": Hold.VALID,
        "task-3": Hold.SUBMITTED,
        "task-4": Hold.SUBMITTED,
        "task-5": Hold.UNRESOLVED,
    }
    only = campaign.plan(cpu_profile(), retry=["task-2", "task-3"], only=["task-2", "task-3"])
    assert only.decision.selected == only.retry == ("task-2", "task-3")
    assert only.decision.held["task-4"] is Hold.NOT_REQUESTED
    incomplete = campaign.plan(cpu_profile(), retry=Retry.INCOMPLETE)
    assert incomplete.decision.selected == ("task-2", "task-4")
    assert incomplete.decision.held["task-3"] is Hold.SUBMITTED


def test_retry_failed_waits_for_slurm_to_forget_the_job(world: World) -> None:
    campaign = world.create(jobs(2))
    campaign.submit(campaign.plan(cpu_profile(), tasks_per_allocation=1))
    world.fake.finish(1000, exit_code=1)
    world.fake.interrupt(1001)
    plan = campaign.plan(cpu_profile(), retry=Retry.FAILED)
    assert plan.decision.selected == ()
    assert dict(plan.decision.held) == {"task-0": Hold.ACTIVE, "task-1": Hold.ACTIVE}
    with pytest.raises(PlanRefused, match="Slurm still lists earlier work.*: 'task-0'$"):
        campaign.plan(cpu_profile(), retry=["task-0"])
    world.fake.purge(1000)
    world.fake.purge(1001)
    plan = campaign.plan(cpu_profile(), retry=Retry.FAILED)
    assert plan.decision.selected == plan.retry == ("task-0", "task-1")


def test_lost_work_is_retried_only_by_its_explicit_key(world: World) -> None:
    campaign = world.create(jobs(1))
    campaign.submit(campaign.plan(cpu_profile()))
    world.fake.start(1000)
    world.fake.lose(1000)
    world.fake.purge(1000)
    bulk = campaign.plan(cpu_profile(), retry=Retry.FAILED)
    assert bulk.decision.selected == () and bulk.decision.held["task-0"] is Hold.SUBMITTED
    explicit = campaign.plan(cpu_profile(), retry=["task-0"])
    assert explicit.decision.selected == explicit.retry == ("task-0",)
    assert campaign.submit(explicit).receipts[0].job == 1001


@pytest.mark.parametrize(
    ("retry", "message"),
    [
        (["nope"], "unknown Task keys: 'nope'"),
        (["task-1"], "retry needs an earlier accepted attempt: 'task-1'"),
        (["task-0"], "Slurm still lists earlier work; wait for it or cancel it: 'task-0'"),
    ],
)
def test_explicit_retries_are_refused_with_their_reason(
    world: World, retry: list[str], message: str
) -> None:
    campaign = world.create(jobs(2))
    campaign.submit(campaign.plan(cpu_profile(), only=["task-0"]))
    world.fake.finish(1000, exit_code=1)
    with pytest.raises(PlanRefused, match=message) as caught:
        campaign.plan(cpu_profile(), retry=retry)
    assert caught.value.keys == tuple(retry)


def test_one_refusal_lists_every_offending_key(world: World) -> None:
    campaign = world.create(jobs(3), probe=Probe({"task-2"}))
    campaign.submit(campaign.plan(cpu_profile(), only=["task-0"]))
    with pytest.raises(PlanRefused, match="'ghost'.*'task-2'.*'task-1'.*'task-0'") as caught:
        campaign.plan(cpu_profile(), retry=["task-0", "task-1", "task-2", "ghost"])
    assert caught.value.keys == ("ghost", "task-2", "task-1", "task-0")


def test_submit_refuses_a_retry_whose_task_became_live(world: World) -> None:
    campaign = world.create(jobs(1))
    campaign.submit(campaign.plan(cpu_profile()))
    world.fake.finish(1000, exit_code=1)
    world.fake.purge(1000)
    failed = campaign.plan(cpu_profile(), retry=["task-0"])
    world.fake.fail_next("sbatch")
    with pytest.raises(SubmissionInterrupted, match="unresolved"):
        campaign.submit(failed)
    world.clock.advance(minutes=15)
    assert len(campaign.reconcile().not_submitted) == 1
    plan = campaign.plan(cpu_profile(), retry=["task-0"])
    world.resubmit(failed.allocations[0])  # the "failed" sbatch reached Slurm after all
    with pytest.raises(StalePlan, match="Slurm still lists earlier work"):
        campaign.submit(plan)
    assert len(campaign.status(scheduler=False).attempts) == 2
    assert world.fake.count("sbatch") == 3


def test_valid_results_are_never_retried(world: World) -> None:
    world.create(jobs(1))
    campaign = world.open(probe=Probe({"task-0"}))
    world.open().submit(world.open().plan(cpu_profile()))  # planned and submitted without a probe
    world.fake.finish(1000)
    world.fake.purge(1000)
    with pytest.raises(PlanRefused, match="valid results cannot be retried: 'task-0'"):
        campaign.plan(cpu_profile(), retry=["task-0"])


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"retry": "task-0"}, "retry must be a collection"),
        ({"retry": [1]}, "retry must contain only Task key strings"),
        ({"retry": None}, "retry must be a collection"),
        ({"only": "task-0"}, "only must be a collection"),
        ({"only": [1]}, "only must contain only Task key strings"),
    ],
)
def test_malformed_requests_are_configuration_errors(
    world: World, options: dict[str, Any], message: str
) -> None:
    campaign = world.create(jobs(1))
    with pytest.raises(ConfigurationError, match=message):
        campaign.plan(cpu_profile(), **options)


def test_selector_shapes_and_probe_requirements_are_checked(world: World) -> None:
    campaign = world.create(jobs(1))
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
    assert document["format"] == "servatus.plan/3" and document["digest"] == plan.digest
    assert document["selected"] == list(keys(0, 1, 2)) and document["probe_required"] is True
    assert document["continuation"] == {"budget": None, "max_links": None}
    assert document["spent"] == {} and document["extensions"] == {}
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
    assert first.decision.selected == second.decision.selected
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
        ("format", "servatus.plan/1", ConfigurationError, "expected format"),
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


def test_saved_plans_out_of_roster_order_or_over_the_batch_bound_are_refused(
    world: World,
) -> None:
    campaign = world.create(jobs(2))
    document = json.loads(campaign.plan(cpu_profile(), tasks_per_allocation=1).to_json())
    reordered = dict(document, selected=["task-1", "task-0"])
    with pytest.raises(ConfigurationError, match="not in roster order"):
        campaign.load_plan(json.dumps(reordered).encode())
    document["profile"]["target"]["max_allocations_per_submit"] = 1
    with pytest.raises(ConfigurationError, match="exceeds the target's max_allocations_per_submit"):
        campaign.load_plan(json.dumps(document).encode())


@pytest.mark.parametrize(
    "data", [b"", b"[]", b"{", b'{"format": "servatus.plan/3"}', b'{"format": "servatus.plan/2"}']
)
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


def test_validate_groups_shapes_without_a_signal_or_site_options(world: World) -> None:
    campaign = world.create(jobs(8))
    bare = cpu_profile(account=None, max_tasks_per_allocation=1)
    plan = campaign.plan(bare)
    assert len(plan.allocations) == 8
    assert [check.task_count for check in campaign.validate(plan)] == [1]


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
