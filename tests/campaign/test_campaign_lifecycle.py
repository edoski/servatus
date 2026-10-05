"""Campaign lifecycle through the public API: authoring, status, and the end-to-end sweep."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from campaign_world import Probe, World, cpu_profile, jobs

from servatus.campaign import AllocationState, Campaign, Hold, ResultState, Retry, Task
from servatus.errors import (
    Busy,
    ConfigurationError,
    Conflict,
    NotFound,
    SubmissionInterrupted,
    Unavailable,
)


def keys(count: int, prefix: str = "task") -> tuple[str, ...]:
    return tuple(f"{prefix}-{index}" for index in range(count))


def test_sweep_from_creation_through_failure_retry_and_saved_plans(world: World) -> None:
    probe = Probe()
    campaign = world.create(jobs(6), probe=probe)
    plan = campaign.plan(cpu_profile())
    assert plan.selected == keys(6) and dict(plan.held) == {} and plan.deferred == ()
    assert [item.task_keys for item in plan.allocations] == [keys(6)[:3], keys(6)[3:]]
    result = campaign.submit(plan)
    assert result.complete and result.unresolved == result.unattempted == ()
    assert [receipt.job.job_id for receipt in result.receipts] == [1000, 1001]

    world.fake.start(1000)
    world.fake.start(1001)
    world.fake.finish(1000)
    world.fake.finish_step(1001, 0)
    world.fake.finish_step(1001, 2)
    world.fake.finish_step(1001, 1, "FAILED", exit_code="1:0")
    world.fake.finish(1001, "FAILED", exit_code="1:0")
    probe.valid.update({"task-0", "task-1", "task-2", "task-3"})
    status = campaign.status()
    counts = status.counts()
    assert (counts["tasks"], counts["valid"], counts["missing"]) == (6, 4, 2)
    assert (counts["succeeded"], counts["failed"]) == (5, 1)
    assert status.quiescent and not status.results_ready
    assert status.tasks[4].exit_code == "1:0"

    retry = campaign.plan(cpu_profile(), retry=Retry.FAILED)
    assert retry.selected == retry.retry == ("task-4",)
    assert dict(retry.held) == {
        **dict.fromkeys(keys(4), Hold.VALID),
        "task-5": Hold.SUBMITTED,
    }
    restored = campaign.load_plan(retry.to_json())
    assert restored == retry
    assert campaign.submit(restored).receipts[0].job.job_id == 1002
    world.fake.finish(1002)

    incomplete = campaign.plan(cpu_profile(), retry=Retry.INCOMPLETE)
    assert incomplete.selected == ("task-4", "task-5")
    probe.valid.update({"task-4", "task-5"})
    final = campaign.status()
    assert final.results_ready and final.quiescent
    assert campaign.plan(cpu_profile(), retry=Retry.FAILED).selected == ()


def test_probe_is_called_once_per_operation_with_only_the_tasks_that_matter(world: World) -> None:
    probe = Probe()
    campaign = world.create(jobs(3), probe=probe)
    campaign.plan(cpu_profile(), only=["task-1"])
    assert probe.calls == [("task-1",)]  # only the requested Tasks can change the decision
    probe.calls.clear()
    plan = campaign.plan(cpu_profile())
    assert probe.calls == [keys(3)]
    campaign.submit(plan)
    assert probe.calls[1:] == [keys(3)]
    campaign.status()
    assert probe.calls[2:] == [keys(3)]
    calls = len(probe.calls)
    assert campaign.plan(cpu_profile()).selected == ()
    assert len(probe.calls) == calls  # accepted work without a retry request is not probed
    world.fake.start(1000)
    world.fake.finish(1000, "FAILED", exit_code="1:0")
    campaign.plan(cpu_profile(), retry=["task-1"])
    assert probe.calls[-1] == ("task-1",)


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        ("task-0", "collection of Task keys"),
        (7, "collection of Task keys"),
        ({1}, "Task key strings"),
        ({"elsewhere"}, "not asked about"),
    ],
)
def test_probe_answers_are_validated(world: World, answer: object, message: str) -> None:
    campaign = world.create(jobs(1), probe=lambda _tasks: answer)  # pyright: ignore[reportArgumentType, reportUnknownLambdaType]
    with pytest.raises(ConfigurationError, match=message):
        campaign.status(scheduler=False)


def test_create_open_and_constructor_are_explicit(world: World) -> None:
    with pytest.raises(NotFound, match="does not exist"):
        world.open()
    campaign = world.create(jobs(2))
    assert world.open().id == campaign.id and len(campaign.id) == 32
    assert world.open().tasks() == jobs(2)
    with pytest.raises(Conflict, match="already exists"):
        world.create(jobs(1))
    with pytest.raises(TypeError, match="Campaign.create"):
        Campaign(object(), None, None, None, None)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ConfigurationError, match="Task keys must be unique"):
        Campaign.create(world.root / "other", jobs(1) + jobs(1))
    with pytest.raises(ConfigurationError, match="probe must be callable"):
        Campaign.open(world.path, probe=1)  # pyright: ignore[reportArgumentType]


def test_ensure_creates_parents_appends_unseen_tasks_and_refuses_changed_ones(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested" / "deeper" / "campaign"
    first = Campaign.ensure(path, jobs(2))
    assert first.tasks() == jobs(2) and not first.status(scheduler=False).sealed
    again = Campaign.ensure(path, jobs(3))
    assert again.id == first.id and again.tasks() == jobs(3)
    revision = again.status(scheduler=False).revision
    Campaign.ensure(path, reversed(jobs(3)))
    assert again.status(scheduler=False).revision == revision
    changed = (*jobs(1), Task("task-1", ("/bin/other",)), *jobs(4)[2:])
    with pytest.raises(Conflict, match="'task-1' differs"):
        Campaign.ensure(path, changed)
    assert again.tasks() == jobs(3)

    sealed = Campaign.ensure(tmp_path / "sealed", jobs(1), appendable=False)
    assert sealed.status(scheduler=False).sealed
    assert Campaign.ensure(tmp_path / "sealed", jobs(1)).tasks() == jobs(1)
    with pytest.raises(Conflict, match="sealed campaign roster cannot change"):
        Campaign.ensure(tmp_path / "sealed", jobs(2))


def test_authoring_appends_only_new_keys_and_seal_is_idempotent(world: World) -> None:
    fixed = Campaign.create(world.root / "fixed", jobs(1))
    with pytest.raises(Conflict, match="sealed"):
        fixed.append(jobs(2)[1:])
    growing = world.create(jobs(1), appendable=True)
    growing.append(jobs(3)[1:])
    assert world.open().tasks() == jobs(3)
    with pytest.raises(Conflict, match="already exist: task-0"):
        growing.append(jobs(1))
    growing.seal()
    before = (world.path / "campaign.json").read_bytes()
    growing.seal()
    assert (world.path / "campaign.json").read_bytes() == before


def test_status_projects_each_task_from_its_own_step(world: World) -> None:
    campaign = world.create(jobs(3))
    campaign.submit(campaign.plan(cpu_profile()))
    world.fake.start(1000)
    world.fake.finish_step(1000, 1, "FAILED", exit_code="2:0")
    status = campaign.status()
    assert [task.execution for task in status.tasks] == [
        AllocationState.RUNNING,
        AllocationState.FAILED,
        AllocationState.RUNNING,
    ]
    assert status.tasks[1].exit_code == "2:0"
    assert not status.quiescent
    attempt = status.attempts[0]
    assert attempt.scheduler is not None and attempt.scheduler.state is AllocationState.RUNNING
    assert [step and step.state for step in attempt.steps] == [
        AllocationState.RUNNING,
        AllocationState.FAILED,
        AllocationState.RUNNING,
    ]


def test_requeue_within_the_submission_window_does_not_wedge_the_task(world: World) -> None:
    campaign = world.create(jobs(1))
    campaign.submit(campaign.plan(cpu_profile()))
    world.fake.start(1000)
    world.fake.requeue(1000)
    assert campaign.status().tasks[0].execution is AllocationState.QUEUED
    world.fake.start(1000)
    world.fake.finish(1000, "FAILED", exit_code="1:0")
    status = campaign.status()
    assert status.tasks[0].execution is AllocationState.FAILED and status.quiescent
    assert campaign.plan(cpu_profile(), retry=Retry.FAILED).selected == ("task-0",)


def test_latest_accepted_attempt_projects_the_task_but_quiescence_uses_all(world: World) -> None:
    campaign = world.create(jobs(1))
    campaign.submit(campaign.plan(cpu_profile()))
    world.fake.finish(1000, "FAILED", exit_code="1:0")
    campaign.submit(campaign.plan(cpu_profile(), retry=Retry.FAILED))
    world.fake.finish(1001)
    world.clock.advance(minutes=5)
    world.fake.requeue(1000, state="RUNNING")
    status = campaign.status()
    assert status.tasks[0].current_allocation_id == status.attempts[1].allocation_id
    assert status.tasks[0].execution is AllocationState.SUCCEEDED
    assert not status.quiescent  # the requeued earlier attempt still holds resources
    world.fake.finish(1000, "FAILED", exit_code="1:0")
    assert campaign.status().quiescent


def test_unresolved_acceptance_dominates_the_task(world: World) -> None:
    campaign = world.create(jobs(2))
    world.wire.on("sbatch", lambda _argv: (_ for _ in ()).throw(Unavailable("reset")))
    plan = campaign.plan(cpu_profile())
    with pytest.raises(SubmissionInterrupted, match="acceptance is unresolved"):
        campaign.submit(plan)
    status = campaign.status()
    assert all(task.unresolved and task.execution is None for task in status.tasks)
    assert status.counts()["unresolved"] == 2 and not status.quiescent


def test_status_offline_makes_no_scheduler_calls_and_json_is_private(world: World) -> None:
    campaign = world.create(jobs(2))
    campaign.submit(campaign.plan(cpu_profile()))
    calls = world.scheduler_calls()
    offline = campaign.status(scheduler=False)
    assert world.scheduler_calls() == calls and not offline.scheduler_observed
    assert [task.result for task in offline.tasks] == [ResultState.UNOBSERVED] * 2
    data = campaign.status().to_json()
    document = json.loads(data)
    assert document["format"] == "servatus.status/1"
    assert [task["key"] for task in document["tasks"]] == ["task-0", "task-1"]
    assert all(secret not in data for secret in (b"secret-0", b"train.py", b"SEED"))


def test_a_concurrent_change_during_observation_is_busy(world: World) -> None:
    other = world.create(jobs(1), appendable=True)

    def appending(tasks: object) -> set[str]:
        other.append([Task(f"late-{len(other.tasks())}", ("/bin/true",))])
        return set()

    campaign = world.open(probe=appending)
    with pytest.raises(Busy, match="changed while it was being observed"):
        campaign.status(scheduler=False)
    with pytest.raises(Busy, match="changed while it was being observed"):
        campaign.plan(cpu_profile())
