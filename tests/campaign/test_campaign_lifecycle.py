"""Campaign lifecycle through the public API: authoring, status, and the end-to-end sweep."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from campaign_world import Probe, World, cpu_profile, jobs

from servatus.campaign import (
    AcceptanceState,
    Campaign,
    Execution,
    Hold,
    Observation,
    ResultState,
    Retry,
    Status,
    Task,
    ping,
    to_document,
)
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


def executions(status: Status) -> list[tuple[Execution | AcceptanceState, int | None]]:
    return [(task.execution, task.exit_code) for task in status.tasks]


def test_ping_and_documents_for_results(world: World) -> None:
    target = cpu_profile().target
    assert ping(target, connect=world.wire) == "slurm 23.11.4"
    with pytest.raises(ConfigurationError, match="ping needs a Target"):
        ping("login.example.edu", connect=world.wire)  # pyright: ignore[reportArgumentType]
    world.wire.down.add(target.host)
    with pytest.raises(Unavailable, match="Connection refused"):
        ping(target, connect=world.wire)
    world.wire.down.clear()
    campaign = world.create(jobs(1))
    result = campaign.submit(campaign.plan(cpu_profile()))
    assert to_document(result) == {
        "receipts": [
            {
                "allocation_id": result.receipts[0].allocation_id,
                "job": 1000,
                "task_keys": ["task-0"],
            }
        ],
        "unresolved": [],
        "unattempted": [],
        "stop_reason": None,
    }


def test_sweep_from_creation_through_failure_retry_and_saved_plans(world: World) -> None:
    probe = Probe()
    campaign = world.create(jobs(6), probe=probe)
    plan = campaign.plan(cpu_profile())
    assert (
        plan.decision.selected == keys(6)
        and dict(plan.decision.held) == {}
        and plan.decision.deferred == ()
    )
    assert [item.task_keys for item in plan.allocations] == [keys(6)[:3], keys(6)[3:]]
    result = campaign.submit(plan)
    assert result.complete and result.unresolved == result.unattempted == ()
    assert [receipt.job for receipt in result.receipts] == [1000, 1001]

    world.fake.start(1000)
    world.fake.start(1001)
    world.fake.finish(1000)
    world.fake.finish_step(1001, 0)
    world.fake.finish_step(1001, 2)
    world.fake.finish(1001, exit_code=1)
    probe.valid.update({"task-0", "task-1", "task-2", "task-3"})
    settling = campaign.status()
    counts = settling.counts()
    assert (counts["tasks"], counts["valid"], counts["missing"]) == (6, 4, 2)
    assert (counts["succeeded"], counts["failed"]) == (5, 1)
    assert [task.exit_code for task in settling.tasks] == [0, 0, 0, 0, 1, 0]
    assert not settling.quiescent and not settling.results_ready
    held = campaign.plan(cpu_profile(), retry=Retry.FAILED)
    assert held.decision.selected == () and held.decision.held["task-4"] is Hold.ACTIVE

    world.fake.purge(1000)
    world.fake.purge(1001)
    assert campaign.status().quiescent
    retry = campaign.plan(cpu_profile(), retry=Retry.FAILED)
    assert retry.decision.selected == retry.retry == ("task-4",)
    assert dict(retry.decision.held) == {
        **dict.fromkeys(keys(4), Hold.VALID),
        "task-5": Hold.SUBMITTED,
    }
    restored = campaign.load_plan(retry.to_json())
    assert restored == retry
    assert campaign.submit(restored).receipts[0].job == 1002
    world.fake.finish(1002)

    incomplete = campaign.plan(cpu_profile(), retry=Retry.INCOMPLETE)
    assert incomplete.decision.selected == ("task-5",)
    assert incomplete.decision.held["task-4"] is Hold.ACTIVE
    world.fake.purge(1002)
    incomplete = campaign.plan(cpu_profile(), retry=Retry.INCOMPLETE)
    assert incomplete.decision.selected == ("task-4", "task-5")
    probe.valid.update({"task-4", "task-5"})
    final = campaign.status()
    assert final.results_ready and final.quiescent
    assert campaign.plan(cpu_profile(), retry=Retry.FAILED).decision.selected == ()


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
    assert campaign.plan(cpu_profile()).decision.selected == ()
    assert len(probe.calls) == calls  # accepted work without a retry request is not probed
    world.fake.finish(1000, exit_code=1)
    world.fake.purge(1000)
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


def test_status_follows_one_allocation_from_queue_to_purge(world: World) -> None:
    campaign = world.create(jobs(3))
    (receipt,) = campaign.submit(campaign.plan(cpu_profile())).receipts
    queued = campaign.status()
    assert executions(queued) == [(Execution.QUEUED, None)] * 3
    assert queued.counts()["queued"] == 3 and not queued.quiescent

    world.fake.start(receipt.job)
    world.fake.finish_step(receipt.job, 1, exit_code=2)
    running = campaign.status()
    assert executions(running) == [
        (Execution.RUNNING, None),
        (Execution.FAILED, 2),
        (Execution.RUNNING, None),
    ]
    assert running.attempts[0].observation == Observation("RUNNING", True, False, (None, 2, None))
    assert not running.quiescent

    world.fake.finish(receipt.job)
    settling = campaign.status()
    assert executions(settling) == [
        (Execution.SUCCEEDED, 0),
        (Execution.FAILED, 2),
        (Execution.SUCCEEDED, 0),
    ]
    assert not settling.quiescent  # Slurm still lists the settling job

    world.fake.purge(receipt.job)
    purged = campaign.status()
    assert executions(purged) == executions(settling)
    assert purged.attempts[0].observation == Observation(None, True, False, (0, 2, 0))
    assert purged.quiescent and purged.warnings == ()


@pytest.mark.parametrize(
    ("ending", "execution", "detail"),
    [
        ("interrupt", Execution.STOPPED, None),
        ("lose", Execution.LOST, None),
        ("cancel pending", Execution.STOPPED, "never started"),
    ],
)
def test_how_an_allocation_ended_is_read_from_its_run_markers(
    world: World, ending: str, execution: Execution, detail: str | None
) -> None:
    campaign = world.create(jobs(2))
    (receipt,) = campaign.submit(campaign.plan(cpu_profile())).receipts
    if ending == "cancel pending":
        assert campaign.cancel(tasks=["task-0"]) == (receipt,)
    else:
        world.fake.start(receipt.job)
        world.fake.finish_step(receipt.job, 0)
        if ending == "interrupt":
            world.fake.interrupt(receipt.job, state="TIMEOUT")
        else:
            world.fake.lose(receipt.job)
    settling = campaign.status()
    assert not settling.quiescent
    assert all(task.detail is None for task in settling.tasks)
    world.fake.purge(receipt.job)
    status = campaign.status()
    assert status.quiescent
    second = status.tasks[1]
    assert (second.execution, second.exit_code, second.detail) == (execution, None, detail)
    first = status.tasks[0]
    if ending == "cancel pending":
        assert (first.execution, first.detail) == (Execution.STOPPED, "never started")
    else:
        assert (first.execution, first.exit_code) == (Execution.SUCCEEDED, 0)
    assert status.counts()[execution.value.lower()] == (2 if ending == "cancel pending" else 1)


def test_quiescence_waits_for_every_accepted_allocation(world: World) -> None:
    campaign = world.create(jobs(2))
    campaign.submit(campaign.plan(cpu_profile(), tasks_per_allocation=1))
    world.fake.finish(1000)
    world.fake.finish(1001)
    world.fake.purge(1000)
    status = campaign.status()
    assert executions(status) == [(Execution.SUCCEEDED, 0)] * 2
    assert not status.quiescent
    world.fake.purge(1001)
    assert campaign.status().quiescent


def test_the_latest_accepted_attempt_owns_the_task(world: World) -> None:
    campaign = world.create(jobs(1))
    campaign.submit(campaign.plan(cpu_profile()))
    world.fake.finish(1000, exit_code=1)
    world.fake.purge(1000)
    campaign.submit(campaign.plan(cpu_profile(), retry=Retry.FAILED))
    world.fake.start(1001)
    status = campaign.status()
    assert status.tasks[0].current_allocation_id == status.attempts[1].allocation_id
    assert executions(status) == [(Execution.RUNNING, None)]
    assert status.attempts[0].observation == Observation(None, True, False, (1,))


def test_unresolved_acceptance_dominates_the_task(world: World) -> None:
    campaign = world.create(jobs(2))
    world.wire.on("sbatch", lambda _argv: (_ for _ in ()).throw(Unavailable("reset")))
    plan = campaign.plan(cpu_profile())
    with pytest.raises(SubmissionInterrupted, match="acceptance is unresolved"):
        campaign.submit(plan)
    status = campaign.status()
    assert all(task.execution is AcceptanceState.UNRESOLVED for task in status.tasks)
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
