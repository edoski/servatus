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
    JobRef,
    PlannedAllocation,
    Task,
    UnattemptedAllocation,
)
from servatus.campaign._state import State
from servatus.campaign._store import Store
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


def test_only_plans_are_submitted_or_validated(world: World) -> None:
    campaign = world.create(jobs(1))
    plan = campaign.plan(cpu_profile())
    for value in (plan.decision, replace(plan, decision=plan.to_json())):
        with pytest.raises(ConfigurationError, match="plan must be a Plan"):
            campaign.submit(value)  # pyright: ignore[reportArgumentType]
        with pytest.raises(ConfigurationError, match="plan must be a Plan"):
            campaign.validate(value)  # pyright: ignore[reportArgumentType]
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
    assert (
        following.decision.selected == ("task-2",)
        and following.decision.held["task-1"] is Hold.UNRESOLVED
    )


def test_unresolved_allocations_are_recovered_by_reconcile_or_marking(world: World) -> None:
    campaign = world.create(jobs(2))
    plan = campaign.plan(cpu_profile(), tasks_per_allocation=1)
    world.fake.lose_next_reply("sbatch")
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
    with pytest.raises(Conflict, match=r"does not prove job 77 for it \(.*0 jobs"):
        campaign.mark_accepted(never, 77)
    assert campaign.plan(cpu_profile()).decision.selected == ("task-1",)


def test_marking_not_submitted_needs_proof_of_absence(world: World) -> None:
    campaign = world.create(jobs(1))
    plan = campaign.plan(cpu_profile("old", host="old.example.edu"))
    allocation = plan.allocations[0].allocation_id
    world.fake.lose_next_reply("sbatch")
    with pytest.raises(SubmissionInterrupted, match="unresolved"):
        campaign.submit(plan)
    world.wire.hosts.clear()
    with pytest.raises(Conflict, match="Slurm holds job 1000 for allocation .* mark-accepted"):
        campaign.mark_not_submitted(allocation)
    assert world.wire.hosts == ["old.example.edu"]  # asked the original target
    world.fake.forget(1000)
    world.wire.on("squeue", reset)
    with pytest.raises(Unavailable, match="connection reset by peer"):
        campaign.mark_not_submitted(allocation)
    world.wire.on("sacct", lambda _argv: Completed(0, b"malformed\n", b""))
    with pytest.raises(ReconciliationError, match="malformed"):
        campaign.mark_not_submitted(allocation)
    assert campaign.status(scheduler=False).attempts[0].acceptance is AcceptanceState.UNRESOLVED
    campaign.mark_not_submitted(allocation)  # the job aged out: absence is proven
    calls = world.scheduler_calls()
    campaign.mark_not_submitted(allocation)  # identical repeat, no contact
    assert world.scheduler_calls() == calls
    assert campaign.status(scheduler=False).attempts[0].acceptance is (
        AcceptanceState.NOT_SUBMITTED
    )


def test_a_proven_job_overrides_a_not_submitted_outcome(world: World) -> None:
    """The receipt commit lost to a concurrent mark-not-submitted; mark-accepted repairs it."""
    campaign = world.create(jobs(2))
    plan = campaign.plan(cpu_profile(), tasks_per_allocation=1)
    first = plan.allocations[0].allocation_id
    world.fake.lose_next_reply("sbatch")
    with pytest.raises(SubmissionInterrupted, match="unresolved"):
        campaign.submit(plan)
    world.wire.on("squeue", lambda _argv: Completed(0, b"", b""))  # Slurm lags behind
    world.wire.on("sacct", lambda _argv: Completed(0, b"", b""))
    campaign.mark_not_submitted(first)
    revision = campaign.status(scheduler=False).revision
    with pytest.raises(Conflict, match=r"does not prove job 1001 for it \(Slurm holds job 1000\)"):
        campaign.mark_accepted(first, 1001)
    with pytest.raises(Conflict, match=r"job 1000;alpha .*Slurm holds job 1000\)"):
        campaign.mark_accepted(first, 1000, cluster="alpha")
    receipt = campaign.mark_accepted(first, 1000)
    assert receipt.job == JobRef(1000) and receipt.task_keys == ("task-0",)
    attempt = campaign.status(scheduler=False).attempts[0]
    assert attempt.acceptance is AcceptanceState.ACCEPTED and attempt.job == JobRef(1000)
    assert campaign.status(scheduler=False).revision == revision + 1
    assert campaign.mark_accepted(first, 1000) == receipt  # identical repeat
    following = campaign.plan(cpu_profile())
    assert following.decision.selected == ("task-1",)


def test_a_not_submitted_outcome_with_later_attempts_cannot_be_overridden(world: World) -> None:
    campaign = world.create(jobs(1))
    world.fake.lose_next_reply("sbatch")
    with pytest.raises(SubmissionInterrupted, match="unresolved") as caught:
        campaign.submit(campaign.plan(cpu_profile()))
    lost = caught.value.result.unresolved[0].allocation_id
    world.wire.on("squeue", lambda _argv: Completed(0, b"", b""))
    world.wire.on("sacct", lambda _argv: Completed(0, b"", b""))
    campaign.mark_not_submitted(lost)
    later = campaign.submit(campaign.plan(cpu_profile())).receipts[0]
    with pytest.raises(Conflict, match=f"later allocations {later.allocation_id} already"):
        campaign.mark_accepted(lost, 1000)


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
    assert [receipt.job for receipt in result.receipts] == [JobRef(1000)]
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


def test_an_identical_concurrent_receipt_does_not_stop_the_batch(world: World) -> None:
    campaign = world.create(jobs(2))
    plan = campaign.plan(cpu_profile(), tasks_per_allocation=1)
    first = plan.allocations[0].allocation_id

    def operator(_argv: tuple[str, ...]) -> None:
        world.open().mark_accepted(first, 1000)

    world.wire.on("sbatch", operator)
    result = campaign.submit(plan)
    assert result.complete and [receipt.job for receipt in result.receipts] == [
        JobRef(1000),
        JobRef(1001),
    ]


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
