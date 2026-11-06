"""Submission through the public API: verification, intent, interruption, and recovery."""

from __future__ import annotations

import errno
import fcntl
import os
import stat
from collections.abc import Callable, Sequence
from dataclasses import replace

import pytest
from campaign_world import Probe, World, cpu_profile, jobs

from servatus.campaign import (
    AcceptanceState,
    Completed,
    Hold,
    PlannedAllocation,
    Receipt,
    ReconcileResult,
    Task,
    UnattemptedAllocation,
)
from servatus.campaign._state import State
from servatus.campaign._store import Store
from servatus.errors import (
    ConfigurationError,
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
    assert [attempt.job for attempt in attempts] == [1000, 1001]
    assert [job.argv for job in world.fake.jobs] == [item.argv for item in plan.allocations]
    assert [job.script for job in world.fake.jobs] == [item.script for item in plan.allocations]


def test_only_plans_are_submitted_or_validated(world: World) -> None:
    campaign = world.create(jobs(1))
    plan = campaign.plan(cpu_profile())
    for value in (plan.decision, replace(plan, decision=plan.to_json())):
        with pytest.raises(ConfigurationError, match="plan must be a Plan"):
            campaign.submit(value)  # pyright: ignore[reportArgumentType]
        with pytest.raises(ConfigurationError, match="plan must be a Plan"):
            campaign.validate(value)  # pyright: ignore[reportArgumentType]
    unencodable = replace(plan, decision=replace(plan.decision, tasks_per_allocation=1.5))
    with pytest.raises(ConfigurationError, match="invalid plan: cannot encode"):
        campaign.submit(unencodable)  # pyright: ignore[reportArgumentType]
    assert world.scheduler_calls() == 0


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
    world.fake.fail_next("ping")
    with pytest.raises(Unavailable, match="injected failure of ping"):
        campaign.submit(plan)
    world.wire.down.add("login.example.edu")
    with pytest.raises(Unavailable, match="Connection refused"):
        campaign.submit(plan)
    assert campaign.status(scheduler=False).attempts == () and world.fake.jobs == ()
    world.wire.down.clear()
    assert campaign.submit(plan).complete


@pytest.mark.parametrize("fault", ["transport", "nonzero", "receipt", "cluster"])
def test_partial_submission_reports_every_allocation(world: World, fault: str) -> None:
    campaign = world.create(jobs(3))
    plan = campaign.plan(cpu_profile(), tasks_per_allocation=1)
    replies = {
        "nonzero": Completed(1, b"", b"sbatch: error: site rejection\n"),
        "receipt": Completed(0, b"not a receipt\n", b""),
        "cluster": Completed(0, b"1001;alpha\n", b""),
    }
    world.wire.on("sbatch", lambda _argv: None)
    world.wire.on("sbatch", reset if fault == "transport" else lambda _argv: replies[fault])
    with pytest.raises(SubmissionInterrupted, match="acceptance is unresolved") as caught:
        campaign.submit(plan)
    result = caught.value.result
    assert isinstance(caught.value.__cause__, Unavailable) and not result.complete
    assert [receipt.job for receipt in result.receipts] == [1000]
    assert [item.allocation_id for item in result.unresolved] == [plan.allocations[1].allocation_id]
    assert result.unresolved[0].observed_job is None
    assert result.unattempted == unattempted(plan.allocations[2])
    if fault == "nonzero":
        assert "site rejection" in str(caught.value)
    assert len(world.fake.jobs) == 1  # nothing reached the queue after the failure
    attempts = campaign.status(scheduler=False).attempts
    assert [attempt.acceptance for attempt in attempts] == [
        AcceptanceState.ACCEPTED,
        AcceptanceState.UNRESOLVED,
    ]
    following = campaign.plan(cpu_profile())
    assert (
        following.decision.selected == ("task-2",)
        and following.decision.held["task-1"] is Hold.UNRESOLVED
    )


def test_a_lost_reply_is_unresolved_until_reconcile_records_the_job(world: World) -> None:
    campaign = world.create(jobs(2))
    plan = campaign.plan(cpu_profile(), tasks_per_allocation=1)
    world.fake.lose_next_reply("sbatch")
    with pytest.raises(SubmissionInterrupted, match="acceptance is unresolved") as caught:
        campaign.submit(plan)
    result = caught.value.result
    lost = plan.allocations[0].allocation_id
    assert [item.allocation_id for item in result.unresolved] == [lost]
    assert result.unattempted == unattempted(plan.allocations[1])
    assert world.fake.job(1000).name == f"servatus-{lost}"
    held = campaign.plan(cpu_profile())
    assert held.decision.selected == ("task-1",)
    assert held.decision.held["task-0"] is Hold.UNRESOLVED
    receipt = Receipt(lost, 1000, ("task-0",))
    assert campaign.reconcile() == ReconcileResult(receipts=(receipt,))
    following = campaign.plan(cpu_profile())
    assert following.decision.held["task-0"] is Hold.SUBMITTED
    assert campaign.submit(following).receipts[0].job == 1001


def test_a_receipt_that_became_durable_despite_a_failed_commit_is_reported(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = world.create(jobs(2))
    plan = campaign.plan(cpu_profile(), tasks_per_allocation=1)
    real_update = Store.update
    commits: list[int] = []

    def update(store: Store, change: Callable[[State], State]) -> State:
        state = real_update(store, change)
        commits.append(state.revision)
        if len(commits) == 2:  # the first receipt is durable, then the commit "fails"
            raise OSError(errno.EIO, "directory sync failed")
        return state

    monkeypatch.setattr(Store, "update", update)
    with pytest.raises(SubmissionInterrupted, match="receipt failed after it became durable") as e:
        campaign.submit(plan)
    result = e.value.result
    assert [receipt.job for receipt in result.receipts] == [1000]
    assert result.unresolved == () and result.unattempted == unattempted(plan.allocations[1])


def test_an_unreadable_campaign_after_a_failed_intent_counts_as_unresolved(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = world.create(jobs(1))
    plan = campaign.plan(cpu_profile())

    def update(_store: Store, _change: Callable[[State], State]) -> State:
        def unreadable(_store: Store) -> State:
            raise OSError(errno.EIO, "read failed")

        monkeypatch.setattr(Store, "read", unreadable)
        raise OSError(errno.EIO, "write failed")

    monkeypatch.setattr(Store, "update", update)
    with pytest.raises(SubmissionInterrupted, match="intent failed after it became durable") as e:
        campaign.submit(plan)
    assert [item.allocation_id for item in e.value.result.unresolved] == [
        plan.allocations[0].allocation_id
    ]
    assert world.fake.jobs == ()


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
    assert isinstance(caught.value.__cause__, ConfigurationError)
    assert isinstance(caught.value.__cause__.__cause__, PermissionError)
    assert result.receipts == ()
    assert result.unresolved[0].observed_job == 1000
    assert result.unattempted == unattempted(plan.allocations[1])
    assert campaign.status(scheduler=False).attempts[0].acceptance is AcceptanceState.UNRESOLVED
    assert campaign.reconcile().receipts == (
        Receipt(plan.allocations[0].allocation_id, 1000, ("task-0",)),
    )


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
    assert [receipt.job for receipt in result.receipts] == [1000]
    assert result.unresolved == () and result.unattempted == unattempted(plan.allocations[1])
    status = campaign.status(scheduler=False)
    assert status.revision == 3 and status.attempts[0].job == 1000


def test_an_identical_concurrent_receipt_does_not_stop_the_batch(world: World) -> None:
    campaign = world.create(jobs(2))
    plan = campaign.plan(cpu_profile(), tasks_per_allocation=1)
    first = plan.allocations[0]

    def reconciled_meanwhile(argv: tuple[str, ...]) -> Completed:
        reply = world.fake.run(argv, stdin=first.script)
        assert world.open().reconcile().receipts[0].job == 1000
        return reply

    world.wire.on("sbatch", reconciled_meanwhile)
    result = campaign.submit(plan)
    assert result.complete and [receipt.job for receipt in result.receipts] == [1000, 1001]


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
    assert (
        following.decision.selected == () and following.decision.held["task-0"] is Hold.UNRESOLVED
    )


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
    assert (world.fake.count("ping"), world.fake.count("sbatch")) == (1, 0)
    assert checked.status(scheduler=False).attempts == ()


def test_a_probe_added_later_is_honoured(world: World) -> None:
    plan = world.create(jobs(1)).plan(cpu_profile())
    probed = world.open(probe=Probe({"task-0"}))
    with pytest.raises(StalePlan, match="no longer eligible"):
        probed.submit(plan)
    assert probed.status(scheduler=False).attempts == ()


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
