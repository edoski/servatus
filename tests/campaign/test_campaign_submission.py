"""Submission through the public API: verification, intent, interruption, and recovery."""

from __future__ import annotations

import errno
import fcntl
import os
import stat
from collections.abc import Sequence
from dataclasses import replace

import pytest
from campaign_world import Probe, World, cpu_profile, jobs

from servatus.campaign import (
    AcceptanceState,
    Completed,
    Hold,
    JobRef,
    PlannedAllocation,
    Task,
    UnattemptedAllocation,
)
from servatus.errors import (
    ConfigurationError,
    Conflict,
    ReconciliationError,
    StalePlan,
    SubmissionInterrupted,
    Unavailable,
)


def unattempted(*allocations: PlannedAllocation) -> tuple[UnattemptedAllocation, ...]:
    return tuple(UnattemptedAllocation(item.allocation_id, item.task_keys) for item in allocations)


def reset(_argv: tuple[str, ...]) -> Completed:
    raise Unavailable("connection reset by peer")


def test_submission_records_intent_before_contact_and_receipts_after(world: World) -> None:
    campaign = world.create(jobs(2))
    plan = campaign.plan(cpu_profile(), tasks_per_allocation=1)
    seen: list[AcceptanceState] = []

    def durable(_argv: tuple[str, ...]) -> None:
        seen.append(world.open().status(scheduler=False).attempts[-1].acceptance)

    world.wire.on("sbatch", durable)
    world.wire.on("sbatch", durable)
    result = campaign.submit(plan)
    assert seen == [AcceptanceState.UNRESOLVED] * 2
    assert [receipt.allocation_id for receipt in result.receipts] == [
        item.allocation_id for item in plan.allocations
    ]
    attempts = campaign.status(scheduler=False).attempts
    assert [attempt.job for attempt in attempts] == [JobRef(1000), JobRef(1001)]
    assert [job.argv for job in world.fake.jobs] == [item.argv for item in plan.allocations]
    assert [job.script for job in world.fake.jobs] == [item.script for item in plan.allocations]


def test_tampered_in_memory_plans_are_refused_before_any_intent(world: World) -> None:
    campaign = world.create(jobs(2))
    plan = campaign.plan(cpu_profile())
    other = cpu_profile("elsewhere", host="other.example.edu", partitions=("debug",))
    item = plan.allocations[0]
    tampered = (
        replace(plan, decision=replace(plan.decision, profile=other)),
        replace(plan, allocations=(replace(item, script=item.script + b"rm -rf ~\n"),)),
        replace(plan, allocations=(replace(item, argv=(*item.argv, "--qos=high")),)),
        replace(plan, digest="0" * 64),
        replace(
            plan, decision=replace(plan.decision, selected=("task-0",), held={"task-1": Hold.VALID})
        ),
    )
    for edited in tampered:
        with pytest.raises(ConfigurationError, match="does not match its decision"):
            campaign.submit(edited)
        with pytest.raises(ConfigurationError, match="does not match its decision"):
            campaign.validate(edited)
    assert world.scheduler_calls() == 0
    assert campaign.status(scheduler=False).attempts == ()
    assert campaign.submit(plan).complete


def test_failed_ping_records_no_intent(world: World) -> None:
    campaign = world.create(jobs(1))
    plan = campaign.plan(cpu_profile())
    world.fake.fail_next("sbatch")
    with pytest.raises(Unavailable, match="injected failure of sbatch"):
        campaign.submit(plan)
    world.wire.down.add("login.example.edu")
    with pytest.raises(Unavailable, match="Connection refused"):
        campaign.submit(plan)
    assert campaign.status(scheduler=False).attempts == () and world.fake.jobs == ()
    world.wire.down.clear()
    assert campaign.submit(plan).complete


@pytest.mark.parametrize("fault", ["transport", "nonzero", "receipt"])
def test_partial_submission_reports_every_allocation(world: World, fault: str) -> None:
    campaign = world.create(jobs(3))
    plan = campaign.plan(cpu_profile(), tasks_per_allocation=1)
    replies = {
        "nonzero": Completed(1, b"", b"sbatch: error: site rejection\n"),
        "receipt": Completed(0, b"not a receipt\n", b""),
    }
    world.wire.on("sbatch", lambda _argv: None)
    world.wire.on("sbatch", reset if fault == "transport" else lambda _argv: replies[fault])
    with pytest.raises(SubmissionInterrupted, match="acceptance is unresolved") as caught:
        campaign.submit(plan)
    result = caught.value.result
    assert isinstance(caught.value.__cause__, Unavailable) and not result.complete
    assert [receipt.job for receipt in result.receipts] == [JobRef(1000)]
    assert [item.allocation_id for item in result.unresolved] == [plan.allocations[1].allocation_id]
    assert result.unresolved[0].observed_job is None
    assert result.unattempted == unattempted(plan.allocations[2])
    if fault == "nonzero":
        assert "site rejection" in str(caught.value)
    assert len(world.fake.jobs) == 1  # nothing reached the queue after the failure
    following = campaign.plan(cpu_profile())
    assert following.selected == ("task-2",) and following.held["task-1"] is Hold.UNRESOLVED


def test_unresolved_allocations_are_recovered_by_reconcile_or_marking(world: World) -> None:
    campaign = world.create(jobs(2))
    plan = campaign.plan(cpu_profile(), tasks_per_allocation=1)
    world.wire.on("sbatch", lambda _argv: world.fake.lose_next_reply("sbatch"))
    with pytest.raises(SubmissionInterrupted, match="unresolved") as caught:
        campaign.submit(plan)
    lost = caught.value.result.unresolved[0].allocation_id
    assert world.fake.jobs[0].name == f"servatus-{lost}"  # Slurm did accept it
    receipt = campaign.reconcile(lost)
    assert receipt.job == JobRef(1000) and receipt.task_keys == ("task-0",)
    assert campaign.reconcile(lost) == receipt  # already accepted: nothing to ask

    retry = campaign.plan(cpu_profile())
    world.wire.on("sbatch", reset)
    with pytest.raises(SubmissionInterrupted, match="unresolved") as caught:
        campaign.submit(retry)
    never = caught.value.result.unresolved[0].allocation_id
    with pytest.raises(ReconciliationError, match="0 jobs"):
        campaign.reconcile(never)
    campaign.mark_not_submitted(never)
    campaign.mark_not_submitted(never)  # identical repeat
    with pytest.raises(Conflict, match="recorded as not submitted"):
        campaign.reconcile(never)
    with pytest.raises(Conflict, match="conflicting outcome"):
        campaign.mark_accepted(never, 77)
    assert campaign.plan(cpu_profile()).selected == ("task-1",)


def test_operator_marks_are_idempotent_and_conflicts_are_refused(world: World) -> None:
    campaign = world.create(jobs(1))
    plan = campaign.plan(cpu_profile())
    allocation = plan.allocations[0].allocation_id

    def operator(_argv: tuple[str, ...]) -> None:
        world.open().mark_accepted(allocation, 1000, cluster=None)

    world.wire.on("sbatch", operator)
    assert campaign.submit(plan).complete
    revision = campaign.status(scheduler=False).revision
    assert campaign.mark_accepted(allocation, 1000).job == JobRef(1000)
    assert campaign.status(scheduler=False).revision == revision
    with pytest.raises(Conflict, match="conflicting outcome"):
        campaign.mark_accepted(allocation, 1001)
    with pytest.raises(Conflict, match="conflicting outcome"):
        campaign.mark_not_submitted(allocation)
    with pytest.raises(ConfigurationError, match="job_id must be a positive integer"):
        campaign.mark_accepted(allocation, 0)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_a_receipt_that_cannot_be_recorded_is_reported_with_the_observed_job(
    world: World,
) -> None:
    campaign = world.create(jobs(2))
    plan = campaign.plan(cpu_profile(), tasks_per_allocation=1)

    def read_only(_argv: tuple[str, ...]) -> None:
        world.path.chmod(0o500)

    world.wire.on("sbatch", read_only)
    try:
        with pytest.raises(SubmissionInterrupted, match="recording the receipt failed") as caught:
            campaign.submit(plan)
    finally:
        world.path.chmod(0o700)
    result = caught.value.result
    assert isinstance(caught.value.__cause__, PermissionError)
    assert result.receipts == ()
    assert result.unresolved[0].observed_job == JobRef(1000)
    assert result.unattempted == unattempted(plan.allocations[1])
    assert campaign.status(scheduler=False).attempts[0].acceptance is AcceptanceState.UNRESOLVED
    assert campaign.mark_accepted(plan.allocations[0].allocation_id, 1000).job == JobRef(1000)


@pytest.mark.parametrize("mutation", ["append", "seal"])
def test_a_concurrent_roster_change_stops_the_batch_after_the_receipt(
    world: World, mutation: str
) -> None:
    campaign = world.create(jobs(2), appendable=True)
    plan = campaign.plan(cpu_profile(), tasks_per_allocation=1)

    def change(_argv: tuple[str, ...]) -> None:
        other = world.open()
        if mutation == "append":
            other.append(jobs(3)[2:])
        else:
            other.seal()

    world.wire.on("sbatch", change)
    with pytest.raises(SubmissionInterrupted, match="next intent was not recorded") as caught:
        campaign.submit(plan)
    assert isinstance(caught.value.__cause__, StalePlan)
    result = caught.value.result
    assert [receipt.job for receipt in result.receipts] == [JobRef(1000)]
    assert result.unresolved == () and result.unattempted == unattempted(plan.allocations[1])
    status = campaign.status(scheduler=False)
    assert status.revision == 3 and status.attempts[0].job == JobRef(1000)


def test_a_change_after_the_last_receipt_does_not_interrupt(world: World) -> None:
    campaign = world.create(jobs(1), appendable=True)
    world.wire.on("sbatch", lambda _argv: world.open().append(jobs(2)[1:]))
    assert campaign.submit(campaign.plan(cpu_profile())).complete


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_interrupts_propagate_and_leave_durable_intent(
    world: World, interrupt: type[BaseException]
) -> None:
    campaign = world.create(jobs(1))
    plan = campaign.plan(cpu_profile())

    def stop(_argv: tuple[str, ...]) -> None:
        raise interrupt("operator stop")

    world.wire.on("sbatch", stop)
    with pytest.raises(interrupt, match="operator stop"):
        campaign.submit(plan)
    attempt = campaign.status(scheduler=False).attempts[0]
    assert attempt.acceptance is AcceptanceState.UNRESOLVED
    following = campaign.plan(cpu_profile())
    assert following.selected == () and following.held["task-0"] is Hold.UNRESOLVED


def test_a_plan_made_with_a_probe_needs_one_to_submit(world: World) -> None:
    probe = Probe()
    world.create(jobs(2), probe=probe)
    data = world.open(probe=probe).plan(cpu_profile()).to_json()
    bare = world.open()
    with pytest.raises(ConfigurationError, match="made with a result probe"):
        bare.submit(bare.load_plan(data))
    probe.valid.add("task-1")
    checked = world.open(probe=probe)
    with pytest.raises(StalePlan, match=r"no longer eligible: 'task-1' \(VALID\)"):
        checked.submit(checked.load_plan(data))
    assert probe.calls[-1] == ("task-0", "task-1")
    assert world.fake.count("sbatch") == 1  # only the ping
    assert checked.status(scheduler=False).attempts == ()


def test_a_probe_added_later_is_honoured(world: World) -> None:
    plan = world.create(jobs(1)).plan(cpu_profile())
    probed = world.open(probe=Probe({"task-0"}))
    with pytest.raises(StalePlan, match="no longer eligible"):
        probed.submit(plan)
    assert probed.status(scheduler=False).attempts == ()


def test_retries_rechecked_at_submission_stay_safe(world: World) -> None:
    campaign = world.create(jobs(1))
    campaign.submit(campaign.plan(cpu_profile()))
    world.fake.finish(1000, "FAILED", exit_code="1:0")
    plan = campaign.plan(cpu_profile(), retry=["task-0"])
    world.fake.requeue(1000, state="RUNNING")
    with pytest.raises(StalePlan, match=r"'task-0' \(ACTIVE\)"):
        campaign.submit(plan)
    world.fake.forget(1000)
    with pytest.raises(StalePlan, match="duplicate-risk acknowledgement"):
        campaign.submit(plan)
    risky = campaign.plan(cpu_profile(), retry=["task-0"], allow_duplicate_risk=["task-0"])
    receipt = campaign.submit(risky).receipts[0]
    attempt = campaign.status(scheduler=False).attempts[-1]
    assert attempt.allocation_id == receipt.allocation_id
    assert (attempt.retry, attempt.duplicate_risk) == (("task-0",), ("task-0",))


def test_an_empty_plan_submits_nothing(world: World) -> None:
    campaign = world.create(jobs(1))
    campaign.submit(campaign.plan(cpu_profile()))
    calls = world.scheduler_calls()
    empty = campaign.plan(cpu_profile())
    assert campaign.submit(empty).receipts == () and campaign.validate(empty) == ()
    assert world.scheduler_calls() == calls


def test_an_intent_that_became_durable_despite_a_failed_commit_is_unresolved(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_fcntl, real_fsync = fcntl.fcntl, os.fsync
    full_sync: int | None = getattr(fcntl, "F_FULLFSYNC", None)

    def without_full_sync(descriptor: int, command: int, *args: int) -> object:
        if command == full_sync:
            raise OSError(errno.ENOTSUP, "F_FULLFSYNC disabled")
        return real_fcntl(descriptor, command, *args)

    def directory_sync_fails(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            monkeypatch.undo()
            raise OSError(errno.EIO, "directory sync failed")
        real_fsync(descriptor)

    def probe(tasks: Sequence[Task]) -> set[str]:
        monkeypatch.setattr(fcntl, "fcntl", without_full_sync)
        monkeypatch.setattr(os, "fsync", directory_sync_fails)
        return set()

    world.create(jobs(2))
    campaign = world.open(probe=probe)
    plan = campaign.plan(cpu_profile(), tasks_per_allocation=1)
    with pytest.raises(SubmissionInterrupted, match="after it became durable") as caught:
        campaign.submit(plan)
    result = caught.value.result
    assert [item.allocation_id for item in result.unresolved] == [plan.allocations[0].allocation_id]
    assert result.unattempted == unattempted(plan.allocations[1])
    assert world.fake.jobs == ()
    attempt = world.open().status(scheduler=False).attempts[0]
    assert attempt.acceptance is AcceptanceState.UNRESOLVED
