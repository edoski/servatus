"""Reconciliation, the not-submitted backstop, cancellation, and logs, each routed to the
Attempt's original target."""

from __future__ import annotations

from datetime import timedelta

import pytest
from campaign_world import START, World, cpu_profile, jobs

from servatus.campaign import (
    AcceptanceState,
    Campaign,
    Completed,
    Execution,
    Hold,
    Plan,
    Receipt,
    ReconcileResult,
    Retry,
    UnattemptedAllocation,
    WaitingAllocation,
)
from servatus.errors import (
    ConfigurationError,
    Conflict,
    EvidenceConflict,
    NotFound,
    StalePlan,
    SubmissionInterrupted,
    Unavailable,
)

LOGS = "/cluster/logs/project"


def interrupted(campaign: Campaign, plan: Plan) -> str:
    """Submit ``plan``, which must stop with its first allocation unresolved; its id."""
    with pytest.raises(SubmissionInterrupted, match="acceptance is unresolved") as caught:
        campaign.submit(plan)
    (unresolved,) = caught.value.result.unresolved
    return unresolved.allocation_id


# --- Reconcile ---------------------------------------------------------------------------------


def test_a_lost_sbatch_reply_is_resolved_from_the_queue_row(world: World) -> None:
    campaign = world.create(jobs(2))
    plan = campaign.plan(cpu_profile(), tasks_per_allocation=1)
    world.fake.lose_next_reply("sbatch")
    lost = interrupted(campaign, plan)
    assert world.fake.job(1000).name == f"servatus-{lost}"  # Slurm did accept it
    receipt = Receipt(lost, 1000, ("task-0",))
    assert campaign.reconcile() == ReconcileResult(receipts=(receipt,))
    attempt = campaign.status(scheduler=False).attempts[0]
    assert (attempt.acceptance, attempt.job) == (AcceptanceState.ACCEPTED, 1000)
    revision = world.revision()
    assert campaign.reconcile() == ReconcileResult()
    assert world.revision() == revision
    assert campaign.plan(cpu_profile()).decision.selected == ("task-1",)


def test_a_purged_job_is_resolved_from_its_start_marker(world: World) -> None:
    campaign = world.create(jobs(1))
    plan = campaign.plan(cpu_profile())
    world.fake.lose_next_reply("sbatch")
    lost = interrupted(campaign, plan)
    world.fake.finish(1000, exit_code=4)
    world.fake.purge(1000)
    world.clock.advance(hours=1)  # long past the not-submitted deadline: evidence still wins
    assert campaign.reconcile().receipts == (Receipt(lost, 1000, ("task-0",)),)
    task = campaign.status().tasks[0]
    assert (task.execution, task.exit_code) == (Execution.FAILED, 4)


def test_no_evidence_waits_fifteen_minutes_then_records_not_submitted(world: World) -> None:
    campaign = world.create(jobs(1))
    plan = campaign.plan(cpu_profile())
    world.fake.fail_next("sbatch")
    never = interrupted(campaign, plan)
    revision = world.revision()
    waiting = WaitingAllocation(never, ("task-0",), START + timedelta(minutes=15))
    assert campaign.reconcile() == ReconcileResult(waiting=(waiting,))
    world.clock.advance(minutes=14, seconds=59)
    result = campaign.reconcile()
    assert result.waiting == (waiting,) and not result.complete
    assert world.revision() == revision
    world.clock.advance(seconds=1)
    result = campaign.reconcile()
    assert result == ReconcileResult(not_submitted=(UnattemptedAllocation(never, ("task-0",)),))
    assert result.complete
    attempt = campaign.status(scheduler=False).attempts[0]
    assert attempt.acceptance is AcceptanceState.NOT_SUBMITTED and attempt.job is None
    revision = world.revision()
    assert campaign.reconcile() == ReconcileResult()
    assert world.revision() == revision
    assert campaign.plan(cpu_profile()).decision.selected == ("task-0",)


def test_reconcile_resolves_every_unresolved_allocation_in_one_pass(world: World) -> None:
    campaign = world.create(jobs(3), appendable=True)
    first = campaign.plan(cpu_profile(), only=["task-0"])
    world.fake.fail_next("sbatch")
    old = interrupted(campaign, first)
    world.clock.advance(minutes=20)
    second = campaign.plan(cpu_profile(), only=["task-1"])
    world.fake.lose_next_reply("sbatch")
    found = interrupted(campaign, second)
    third = campaign.plan(cpu_profile(), only=["task-2"])
    world.fake.fail_next("sbatch")
    young = interrupted(campaign, third)
    result = campaign.reconcile()
    assert result.receipts == (Receipt(found, 1000, ("task-1",)),)
    assert result.not_submitted == (UnattemptedAllocation(old, ("task-0",)),)
    assert [item.allocation_id for item in result.waiting] == [young]


def test_two_jobs_for_one_identity_are_an_evidence_conflict(world: World) -> None:
    campaign = world.create(jobs(1))
    plan = campaign.plan(cpu_profile())
    world.fake.lose_next_reply("sbatch")
    lost = interrupted(campaign, plan)
    world.resubmit(plan.allocations[0])
    revision = world.revision()
    with pytest.raises(
        EvidenceConflict, match=f"allocation {lost}: evidence names jobs 1000, 1001"
    ):
        campaign.reconcile()
    world.fake.start(1000)
    world.fake.purge(1000)
    with pytest.raises(EvidenceConflict, match="evidence names jobs 1000, 1001"):
        campaign.reconcile()  # a start marker of one job and the queue row of another
    assert world.revision() == revision


@pytest.mark.parametrize(
    ("command", "reply", "message"),
    [
        ("squeue", None, "injected failure of squeue"),
        ("find", None, "injected failure of find"),
        ("squeue", Completed(1, b"", b"squeue: error: timeout\n"), "squeue failed"),
        ("find", Completed(1, b"", b"find: no such directory\n"), f"listing {LOGS} failed"),
    ],
)
def test_reconcile_records_nothing_when_evidence_is_unavailable(
    world: World, command: str, reply: Completed | None, message: str
) -> None:
    campaign = world.create(jobs(1))
    plan = campaign.plan(cpu_profile())
    world.fake.lose_next_reply("sbatch")
    interrupted(campaign, plan)
    world.clock.advance(minutes=30)
    revision = world.revision()
    world.fake.fail_next(command, reply)
    with pytest.raises(Unavailable, match=message):
        campaign.reconcile()
    assert world.revision() == revision
    assert campaign.status(scheduler=False).attempts[0].acceptance is AcceptanceState.UNRESOLVED


def test_reconcile_asks_the_original_target(world: World) -> None:
    campaign = world.create(jobs(1))
    plan = campaign.plan(cpu_profile("old", host="old.example.edu"))
    world.fake.lose_next_reply("sbatch")
    interrupted(campaign, plan)
    world.wire.hosts.clear()
    assert campaign.reconcile().receipts[0].job == 1000
    assert set(world.wire.hosts) == {"old.example.edu"}


# --- Not-submitted backstop --------------------------------------------------------------------


def test_a_job_behind_a_not_submitted_allocation_holds_its_tasks(world: World) -> None:
    campaign = world.create(jobs(2))
    failed = campaign.plan(cpu_profile())
    world.fake.fail_next("sbatch")
    never = interrupted(campaign, failed)
    world.clock.advance(minutes=15)
    campaign.reconcile()
    fresh = campaign.plan(cpu_profile())
    assert fresh.decision.selected == ("task-0", "task-1")

    job = world.resubmit(failed.allocations[0])
    status = campaign.status()
    (warning,) = status.warnings
    assert f"allocation {never}" in warning and f"job {job}" in warning
    assert not status.quiescent
    assert status.counts()["unsubmitted"] == 2
    held = campaign.plan(cpu_profile())
    assert held.decision.selected == ()
    assert dict(held.decision.held) == {"task-0": Hold.ACTIVE, "task-1": Hold.ACTIVE}
    with pytest.raises(StalePlan, match=r"no longer eligible: 'task-0' \(ACTIVE\)"):
        campaign.submit(fresh)

    assert campaign.cancel(allocations=[never]) == (Receipt(never, job, ("task-0", "task-1")),)
    assert world.fake.job(job).state == "CANCELLED"
    assert campaign.status().warnings == (warning,)  # settling: still listed
    world.fake.purge(job)
    status = campaign.status()
    assert status.warnings == () and status.quiescent
    assert campaign.plan(cpu_profile()).decision.selected == ("task-0", "task-1")


def test_cancel_by_task_reaches_a_job_behind_a_not_submitted_allocation(world: World) -> None:
    campaign = world.create(jobs(1))
    failed = campaign.plan(cpu_profile())
    world.fake.fail_next("sbatch")
    never = interrupted(campaign, failed)
    world.clock.advance(minutes=15)
    campaign.reconcile()
    assert campaign.cancel(tasks=["task-0"]) == ()
    job = world.resubmit(failed.allocations[0])
    assert campaign.cancel(tasks=["task-0"]) == (Receipt(never, job, ("task-0",)),)


def test_evidence_of_another_job_under_an_accepted_identity_is_a_conflict(world: World) -> None:
    campaign = world.create(jobs(1))
    plan = campaign.plan(cpu_profile())
    (receipt,) = campaign.submit(plan).receipts
    world.fake.write_file(f"{LOGS}/{receipt.allocation_id}.999.start")
    with pytest.raises(EvidenceConflict, match="a run marker names job 999, not the accepted"):
        campaign.status()
    world.fake.remove_file(f"{LOGS}/{receipt.allocation_id}.999.start")
    world.resubmit(plan.allocations[0])
    with pytest.raises(EvidenceConflict, match="Slurm lists jobs 1000, 1001"):
        campaign.status()


# --- Cancel ------------------------------------------------------------------------------------


def test_cancel_stops_only_live_allocations_that_are_not_settling(world: World) -> None:
    campaign = world.create(jobs(4))
    first, second = campaign.submit(campaign.plan(cpu_profile(), tasks_per_allocation=2)).receipts
    world.fake.start(1000)
    assert campaign.cancel(tasks=["task-1"]) == (first,)
    assert world.fake.job(1000).state == "CANCELLED"
    assert world.fake.job(1001).state == "PENDING"
    scancels = world.fake.count("scancel")
    assert campaign.cancel(allocations=[first.allocation_id]) == ()  # settling
    world.fake.purge(1000)
    assert campaign.cancel(tasks=["task-0"]) == ()  # gone
    assert world.fake.count("scancel") == scancels
    assert campaign.cancel(allocations=[second.allocation_id], tasks=["task-3"]) == (second,)
    assert world.fake.count("scancel") == scancels + 1
    world.fake.purge(1001)
    status = campaign.status()
    assert [(task.execution, task.detail) for task in status.tasks] == [
        (Execution.STOPPED, None),
        (Execution.STOPPED, None),
        (Execution.STOPPED, "never started"),
        (Execution.STOPPED, "never started"),
    ]
    retry = campaign.plan(cpu_profile(), retry=Retry.FAILED)
    assert retry.decision.selected == ("task-0", "task-1", "task-2", "task-3")


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
    assert [job.state for job in world.fake.jobs] == ["PENDING", "CANCELLED", "PENDING"]


def test_cancel_refuses_unknown_or_unresolved_targets(world: World) -> None:
    campaign = world.create(jobs(2))
    with pytest.raises(ConfigurationError, match="needs Task keys or allocation ids"):
        campaign.cancel()
    with pytest.raises(NotFound, match="unknown Task: 'nope'"):
        campaign.cancel(tasks=["nope"])
    with pytest.raises(Conflict, match="'task-0' has no submitted allocation to cancel"):
        campaign.cancel(tasks=["task-0"])
    with pytest.raises(NotFound, match="unknown allocation"):
        campaign.cancel(allocations=["0" * 24])
    plan = campaign.plan(cpu_profile())
    world.fake.lose_next_reply("sbatch")
    lost = interrupted(campaign, plan)
    with pytest.raises(Conflict, match=f"allocation {lost} is unresolved; reconcile it first"):
        campaign.cancel(allocations=[lost])
    with pytest.raises(Conflict, match="'task-0' has no submitted allocation to cancel"):
        campaign.cancel(tasks=["task-0"])
    assert world.fake.count("scancel") == 0


# --- Logs --------------------------------------------------------------------------------------


def test_logs_are_exact_bounded_suffixes_of_allocation_and_step_logs(world: World) -> None:
    campaign = world.create(jobs(2))
    allocation = campaign.submit(campaign.plan(cpu_profile())).receipts[0].allocation_id
    root = "/cluster/logs/project"
    world.fake.write_file(f"{root}/{allocation}-1000.out", b"\xff\x00batch\x1b[2J")
    world.fake.write_file(f"{root}/{allocation}-1000-1.out", b"0123456789")
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
    world.fake.finish(1000, exit_code=1)
    world.fake.purge(1000)
    world.wire.hosts.clear()
    new = cpu_profile("new", host="new.example.edu", log_root="/new/logs")
    plan = campaign.plan(new, retry=Retry.FAILED)
    assert set(world.wire.hosts) == {"old.example.edu"}
    second = campaign.submit(plan).receipts[0]
    assert world.fake.job(1001).name == f"servatus-{second.allocation_id}"
    world.wire.hosts.clear()
    campaign.status()
    assert set(world.wire.hosts) == {"new.example.edu", "old.example.edu"}
    world.fake.write_file(f"/old/logs/{first.allocation_id}-1000.out", b"old")
    world.fake.write_file(f"/new/logs/{second.allocation_id}-1001.out", b"new")
    world.wire.hosts.clear()
    assert campaign.read_log(allocation=first.allocation_id).content == b"old"
    assert campaign.read_log(allocation=second.allocation_id).content == b"new"
    assert world.wire.hosts == ["old.example.edu", "new.example.edu"]
