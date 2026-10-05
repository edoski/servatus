"""Cancellation, logs, and reconciliation, each routed to the Attempt's original target."""

from __future__ import annotations

import pytest
from campaign_world import World, cpu_profile, jobs

from servatus.campaign import AllocationState, Completed, Hold, JobRef, Retry
from servatus.errors import (
    ConfigurationError,
    Conflict,
    NotFound,
    SubmissionInterrupted,
    Unavailable,
)


def test_cancel_stops_live_allocations_of_tasks_or_ids(world: World) -> None:
    campaign = world.create(jobs(4))
    plan = campaign.plan(cpu_profile(), tasks_per_allocation=2)
    first, second = campaign.submit(plan).receipts
    world.fake.start(1000)
    assert campaign.cancel(tasks=["task-1"]) == (first,)
    assert world.fake.job(1000).state.startswith("CANCELLED")
    assert world.fake.job(1001).state == "PENDING"
    status = campaign.status()
    assert [task.execution for task in status.tasks[:2]] == [AllocationState.CANCELLED] * 2

    scancels = world.fake.count("scancel")
    assert campaign.cancel(allocations=[first.allocation_id]) == ()  # already terminal
    assert world.fake.count("scancel") == scancels
    assert campaign.cancel(allocations=[second.allocation_id], tasks=["task-3"]) == (second,)
    assert world.fake.count("scancel") == scancels + 1
    # The queued allocation never started a step, so its Tasks' own outcomes are unknown.
    retry = campaign.plan(cpu_profile(), retry=Retry.FAILED)
    assert retry.selected == ("task-0", "task-1")
    assert dict(retry.decision.held) == {
        "task-2": Hold.UNOBSERVABLE,
        "task-3": Hold.UNOBSERVABLE,
    }


def test_cancel_reaches_every_live_attempt_of_a_task(world: World) -> None:
    campaign = world.create(jobs(1))
    campaign.submit(campaign.plan(cpu_profile()))
    world.fake.forget(1000)
    campaign.submit(campaign.plan(cpu_profile(), retry=["task-0"], allow_duplicate_risk=["task-0"]))
    receipts = campaign.cancel(tasks=["task-0"])
    assert [receipt.job for receipt in receipts] == [JobRef(1000), JobRef(1001)]


def test_cancel_attempts_every_allocation_and_reports_failures(world: World) -> None:
    campaign = world.create(jobs(3))
    first, second, third = campaign.submit(
        campaign.plan(cpu_profile(), tasks_per_allocation=1)
    ).receipts

    def refused(_argv: tuple[str, ...]) -> Completed:
        return Completed(1, b"", b"scancel: error: Access/permission denied\n")

    world.wire.on("scancel", refused)
    world.wire.on("scancel", lambda _argv: None)
    world.wire.on("scancel", refused)
    with pytest.raises(Unavailable, match="cancel failed for") as caught:
        campaign.cancel(tasks=["task-0", "task-1", "task-2"])
    message = str(caught.value)
    assert f"{first.allocation_id} (scancel failed with exit status 1" in message
    assert f"{third.allocation_id} (scancel" in message
    assert message.endswith(f"cancelled: {second.allocation_id}")
    assert [job.state for job in world.fake.jobs] == ["PENDING", "CANCELLED by 1000", "PENDING"]


def test_cancel_still_stops_terminal_work_the_scheduler_retains(world: World) -> None:
    campaign = world.create(jobs(1))
    (receipt,) = campaign.submit(campaign.plan(cpu_profile())).receipts
    world.fake.start(1000)
    world.fake.finish(1000, in_queue=True, accounted=False)  # accounting still says RUNNING
    evidence = campaign.status().attempts[0].scheduler
    assert evidence is not None and evidence.state.terminal and evidence.retained
    assert campaign.cancel(tasks=["task-0"]) == (receipt,)
    assert world.fake.count("scancel") == 1


def test_cancel_by_task_skips_attempts_that_were_never_submitted(world: World) -> None:
    campaign = world.create(jobs(1))
    world.wire.on("sbatch", reset_sbatch)
    with pytest.raises(SubmissionInterrupted, match="unresolved") as caught:
        campaign.submit(campaign.plan(cpu_profile()))
    campaign.mark_not_submitted(caught.value.result.unresolved[0].allocation_id)
    with pytest.raises(Conflict, match="'task-0' has no accepted allocation to cancel"):
        campaign.cancel(tasks=["task-0"])
    (receipt,) = campaign.submit(campaign.plan(cpu_profile())).receipts
    assert campaign.cancel(tasks=["task-0"]) == (receipt,)


def reset_sbatch(_argv: tuple[str, ...]) -> Completed:
    raise Unavailable("connection reset by peer")


def test_cancel_refuses_unknown_or_unaccepted_targets(world: World) -> None:
    campaign = world.create(jobs(2))
    with pytest.raises(ConfigurationError, match="needs Task keys or allocation ids"):
        campaign.cancel()
    with pytest.raises(NotFound, match="unknown Task: 'nope'"):
        campaign.cancel(tasks=["nope"])
    with pytest.raises(Conflict, match="no accepted allocation to cancel"):
        campaign.cancel(tasks=["task-0"])
    with pytest.raises(NotFound, match="unknown allocation"):
        campaign.cancel(allocations=["0" * 24])
    world.fake.lose_next_reply("sbatch")
    with pytest.raises(SubmissionInterrupted, match="unresolved") as caught:
        campaign.submit(campaign.plan(cpu_profile()))
    allocation = caught.value.result.unresolved[0].allocation_id
    with pytest.raises(Conflict, match="no accepted job"):
        campaign.cancel(allocations=[allocation])
    assert world.fake.count("scancel") == 0


def test_logs_are_exact_bounded_suffixes_of_allocation_and_step_logs(world: World) -> None:
    campaign = world.create(jobs(2))
    allocation = campaign.submit(campaign.plan(cpu_profile())).receipts[0].allocation_id
    root = "/cluster/logs/project"
    world.fake.write_log(f"{root}/{allocation}-1000.out", b"\xff\x00batch\x1b[2J")
    world.fake.write_log(f"{root}/{allocation}-1000-1.out", b"0123456789")
    snapshot = campaign.read_log(allocation=allocation)
    assert (snapshot.content, snapshot.truncated) == (b"\xff\x00batch\x1b[2J", False)
    assert snapshot.observed_at == world.clock.now
    step = campaign.read_log(task="task-1", max_bytes=4)
    assert (step.content, step.truncated) == (b"6789", True)
    assert campaign.read_log(task="task-1", allocation=allocation, max_bytes=10).content == (
        b"0123456789"
    )
    with pytest.raises(Unavailable, match="log is unavailable"):
        campaign.read_log(task="task-0")
    assert "secret" not in repr(snapshot)


@pytest.mark.parametrize("maximum", [0, 1024 * 1024 + 1, True, 1.5])
def test_log_bounds_are_checked_before_contact(world: World, maximum: object) -> None:
    campaign = world.create(jobs(1))
    allocation = campaign.submit(campaign.plan(cpu_profile())).receipts[0].allocation_id
    calls = world.scheduler_calls()
    with pytest.raises(ConfigurationError, match="max_bytes must be an integer from 1 to 1048576"):
        campaign.read_log(allocation=allocation, max_bytes=maximum)  # pyright: ignore[reportArgumentType]
    assert world.scheduler_calls() == calls


def test_log_selection_errors_are_reported_before_contact(world: World) -> None:
    campaign = world.create(jobs(2), appendable=True)
    allocation = (
        campaign.submit(campaign.plan(cpu_profile(), only=["task-0"])).receipts[0].allocation_id
    )
    calls = world.scheduler_calls()
    with pytest.raises(ConfigurationError, match="needs a task, an allocation, or both"):
        campaign.read_log()
    with pytest.raises(NotFound, match="'task-1' has no accepted allocation"):
        campaign.read_log(task="task-1")
    with pytest.raises(NotFound, match="unknown Task: 'ghost'"):
        campaign.read_log(task="ghost")
    with pytest.raises(NotFound, match="'task-1' is not in allocation"):
        campaign.read_log(task="task-1", allocation=allocation)
    with pytest.raises(NotFound, match="unknown allocation"):
        campaign.read_log(allocation="f" * 24)
    assert world.scheduler_calls() == calls


def test_history_keeps_each_attempt_on_its_original_route(world: World) -> None:
    campaign = world.create(jobs(1))
    old = cpu_profile("old", host="old.example.edu", log_root="/old/logs")
    first = campaign.submit(campaign.plan(old)).receipts[0]
    world.fake.finish(1000, "FAILED", exit_code="1:0")
    world.wire.hosts.clear()
    new = cpu_profile("new", host="new.example.edu", log_root="/new/logs")
    second = campaign.submit(campaign.plan(new, retry=Retry.FAILED)).receipts[0]
    assert world.wire.hosts == ["old.example.edu", "new.example.edu", "old.example.edu"]
    world.wire.hosts.clear()
    campaign.status()
    assert sorted(map(str, world.wire.hosts)) == ["new.example.edu", "old.example.edu"]
    world.fake.write_log(f"/old/logs/{first.allocation_id}-1000.out", b"old")
    world.fake.write_log(f"/new/logs/{second.allocation_id}-1001.out", b"new")
    world.wire.hosts.clear()
    assert campaign.read_log(allocation=first.allocation_id).content == b"old"
    assert campaign.read_log(allocation=second.allocation_id).content == b"new"
    assert world.wire.hosts == ["old.example.edu", "new.example.edu"]


def test_reconcile_asks_the_original_target(world: World) -> None:
    campaign = world.create(jobs(1))
    plan = campaign.plan(cpu_profile("old", host="old.example.edu"))
    world.fake.lose_next_reply("sbatch")
    with pytest.raises(SubmissionInterrupted, match="unresolved"):
        campaign.submit(plan)
    world.wire.hosts.clear()
    receipt = campaign.reconcile(plan.allocations[0].allocation_id)
    assert receipt.job == JobRef(1000) and world.wire.hosts == ["old.example.edu"]
    with pytest.raises(NotFound, match="unknown allocation"):
        campaign.reconcile("a" * 24)
