"""Leases and their steps through the public Campaign API, with ``FakeScheduler`` acting as the
lease's batch script and Slurm's step scheduler."""

from __future__ import annotations

import contextlib
import tempfile
from datetime import timedelta
from pathlib import Path
from random import Random
from typing import Any

import pytest
from campaign_world import World, cpu_profile, jobs, make_world
from hypothesis import given, settings
from hypothesis import strategies as st
from support.builders import resources

from servatus.campaign import (
    AcceptanceState,
    Campaign,
    Continuation,
    Execution,
    Hold,
    Profile,
    Receipt,
    Retry,
    StopReason,
    TaskStatus,
)
from servatus.campaign._scheduler import ORDER
from servatus.campaign._state import Kind
from servatus.errors import (
    ConfigurationError,
    Conflict,
    EvidenceConflict,
    NotFound,
    PlanRefused,
    StalePlan,
    SubmissionInterrupted,
    Unavailable,
)

LOGS = "/cluster/logs/project"
MINUTE = timedelta(minutes=1)


def lease_profile(**changes: Any) -> Profile:
    """Four 2-CPU slots fit; a 5-minute pre-limit signal."""
    plain = cpu_profile(**changes)
    limits = resources(
        cpus=2, memory_mib=1024, gpus=0, time_limit="01:00:00", signal_before_end="00:05:00"
    )
    return Profile("lease", plain.target, limits)


def leased(world: World, count: int = 2, *, links: int | None = None) -> tuple[Campaign, list[str]]:
    """A campaign of ``count`` Tasks and a submitted lease of 2 slots; its link ids."""
    campaign = world.create(jobs(count))
    plan = campaign.hold(lease_profile(), slots=2, continuation=Continuation(max_links=links))
    campaign.submit(plan)
    return campaign, [item.allocation_id for item in plan.allocations]


def ordered(campaign: Campaign, lease: str, **options: Any) -> dict[str, str]:
    """Submit a step plan into ``lease``; each started Task's step id."""
    plan = campaign.plan(lease=lease, **options)
    campaign.submit(plan)
    return {item.task_keys[0]: item.allocation_id for item in plan.allocations}


def running(world: World, count: int = 2, **options: Any) -> tuple[Campaign, str, dict[str, str]]:
    """A running lease (job 1000) that took a step of every Task."""
    campaign, (lease, *_) = leased(world, count)
    world.fake.start(1000)
    steps = ordered(campaign, lease, **options)
    assert set(world.fake.poll(1000)) == set(steps.values())
    return campaign, lease, steps


def task(campaign: Campaign, key: str = "task-0") -> TaskStatus:
    return next(item for item in campaign.status().tasks if item.key == key)


def outcome(campaign: Campaign, key: str = "task-0") -> tuple[object, ...]:
    found = task(campaign, key)
    return found.execution, found.exit_code, found.detail


def marker(step: str, suffix: str, job: int = 1000) -> str:
    return f"{LOGS}/{step}.{job}.{suffix}"


# --- holding a lease -------------------------------------------------------------------------


def test_hold_plans_one_lease_of_slots_and_selects_no_task(world: World) -> None:
    campaign = world.create(jobs(3))
    plan = campaign.hold(lease_profile(), slots=3)
    (item,) = plan.allocations
    assert (item.slots, item.within, item.task_keys) == (3, None, ())
    assert (item.cpus, item.memory_mib, item.gpus) == (6, 3072, 0)
    assert "--ntasks=3" in item.argv and item.argv[-1] == "--signal=B:USR1@300"
    assert plan.decision.selected == () and plan.retry == ()
    assert dict(plan.decision.held) == {f"task-{index}": Hold.NOT_REQUESTED for index in range(3)}
    assert campaign.hold(lease_profile()).decision.slots == 4
    assert world.open().load_plan(plan.to_json()) == plan
    assert world.fake.calls == ()


@pytest.mark.parametrize(
    ("profile", "options", "message"),
    [
        (cpu_profile(), {}, "a lease needs resources.signal_before_end"),
        (lease_profile(), {"continuation": Continuation(budget="02:00:00")}, "takes no budget"),
        (lease_profile(), {"slots": 5}, "exceeds the feasible capacity 4"),
        ("l40s", {}, "profile must be a Profile"),
        (lease_profile(), {"continuation": 2}, "continuation must be a Continuation"),
    ],
)
def test_hold_refusals(world: World, profile: Any, options: dict[str, Any], message: str) -> None:
    campaign = world.create(jobs(1))
    with pytest.raises(ConfigurationError, match=message):
        campaign.hold(profile, **options)
    assert world.fake.calls == ()


def test_a_lease_is_submitted_as_one_job_for_no_task(world: World) -> None:
    campaign = world.create(jobs(2))
    plan = campaign.hold(lease_profile(), slots=2)
    result = campaign.submit(plan)
    assert result.receipts == (Receipt(plan.allocations[0].allocation_id, 1000, ()),)
    assert world.fake.job(1000).task_count == 2 and world.fake.job(1000).state == "PENDING"
    assert {item.key: item.current_allocation_id for item in campaign.status().tasks} == {
        "task-0": None,
        "task-1": None,
    }


def test_a_chain_of_leases_follows_its_first_link_and_is_cancelled_newest_first(
    world: World,
) -> None:
    campaign, (first, second) = leased(world, links=2)
    assert world.fake.job(1001).dependencies == (1000,)
    assert campaign.cancel(allocations=[first]) == (
        Receipt(second, 1001, ()),
        Receipt(first, 1000, ()),
    )
    assert [job.state for job in world.fake.jobs] == ["CANCELLED", "CANCELLED"]


# --- steps -----------------------------------------------------------------------------------


def test_steps_are_orders_that_run_once_the_lease_takes_them_and_slurm_finds_room(
    world: World,
) -> None:
    campaign, (lease, *_) = leased(world)
    world.fake.start(1000)
    plan = campaign.plan(lease=lease)
    assert plan.decision.selected == ("task-0", "task-1")
    assert all(
        item.argv == () and (item.slots, item.within) == (1, lease) for item in plan.allocations
    )
    assert dict(plan.decision.spent) == {"task-0": timedelta(0), "task-1": timedelta(0)}
    result = campaign.submit(plan)
    steps = [item.allocation_id for item in plan.allocations]
    assert result.receipts == tuple(
        Receipt(step, 1000, (key,)) for step, key in zip(steps, ("task-0", "task-1"), strict=True)
    )
    assert world.fake.count("sbatch") == 1
    assert outcome(campaign) == (Execution.QUEUED, None, None)
    world.clock.advance(minutes=5)
    assert set(world.fake.poll(1000)) == set(steps)
    assert outcome(campaign) == (Execution.QUEUED, None, None)
    world.clock.advance(minutes=5)
    world.fake.begin_step(steps[0])
    assert outcome(campaign) == (Execution.RUNNING, None, None)
    world.clock.advance(minutes=10)
    world.fake.end_step(steps[0])
    found = task(campaign)
    assert (found.execution, found.exit_code) == (Execution.SUCCEEDED, 0)
    assert found.run_time is not None and found.run_time.total == 10 * MINUTE
    attempt = next(item for item in campaign.status().attempts if item.allocation_id == steps[0])
    assert (attempt.slots, attempt.within, attempt.job) == (1, lease, 1000)


def test_orders_written_while_the_lease_is_queued_are_taken_once_it_starts(
    world: World,
) -> None:
    campaign, (lease, *_) = leased(world)
    steps = ordered(campaign, lease)
    assert outcome(campaign) == (Execution.QUEUED, None, None)
    world.fake.start(1000)
    assert world.fake.poll(1000) == tuple(sorted(steps.values()))


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (0, (Execution.SUCCEEDED, 0, None)),
        (3, (Execution.FAILED, 3, None)),
        (75, (Execution.STOPPED, 75, StopReason.YIELDED)),
    ],
)
def test_a_step_reads_as_its_end_marker_says(
    world: World, code: int, expected: tuple[object, ...]
) -> None:
    campaign, _, steps = running(world)
    world.fake.begin_step(steps["task-0"])
    world.fake.end_step(steps["task-0"], code)
    assert outcome(campaign) == expected
    assert outcome(campaign, "task-1") == (Execution.QUEUED, None, None)


def test_steps_do_not_count_against_max_allocations_per_submit(world: World) -> None:
    profile = lease_profile(max_allocations_per_submit=1)
    campaign = world.create(jobs(3))
    campaign.submit(campaign.hold(profile, slots=1))
    lease = campaign.status().attempts[0].allocation_id
    assert len(campaign.plan(lease=lease).allocations) == 3


def test_steps_start_in_the_first_link_of_a_lease_chain_that_takes_orders(world: World) -> None:
    campaign, (first, second) = leased(world, links=2)
    world.fake.start(1000)
    for named in (first, second):
        assert campaign.plan(lease=named).decision.within == first
    world.fake.signal_limit(1000)
    assert campaign.plan(lease=first).decision.within == second
    world.fake.interrupt(1000, "TIMEOUT")
    world.fake.start(1001)
    assert campaign.plan(lease=first).decision.within == second
    world.fake.interrupt(1001, "TIMEOUT")
    with pytest.raises(Conflict, match="has no queued or running link before its pre-limit"):
        campaign.plan(lease=second)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"profile": lease_profile()}, "takes its Profile from its lease"),
        ({"tasks_per_allocation": 1}, "takes its Profile from its lease"),
        ({"continuation": Continuation(max_links=2)}, "plan without max_links"),
        ({"slots": 3}, "holds only 2 slots"),
        ({"slots": 0}, "slots must be a positive integer"),
    ],
)
def test_step_plan_refusals(world: World, options: dict[str, Any], message: str) -> None:
    campaign, (lease, *_) = leased(world)
    with pytest.raises(ConfigurationError, match=message):
        campaign.plan(lease=lease, **options)


def test_a_step_plan_needs_a_lease_of_the_campaign(world: World) -> None:
    campaign = world.create(jobs(2))
    with pytest.raises(ConfigurationError, match="slots needs a lease"):
        campaign.plan(cpu_profile(), slots=1)
    campaign.submit(campaign.plan(cpu_profile(), only=["task-0"]))
    plain = campaign.status().attempts[0].allocation_id
    with pytest.raises(ConfigurationError, match=f"allocation {plain} is not a lease"):
        campaign.plan(lease=plain)
    with pytest.raises(NotFound, match="unknown allocation"):
        campaign.plan(lease="f" * 24)


@pytest.mark.parametrize("ending", ["signal_limit", "interrupt"])
def test_a_saved_step_plan_goes_stale_when_its_lease_stops_taking_orders(
    world: World, ending: str
) -> None:
    campaign, (lease, *_) = leased(world)
    world.fake.start(1000)
    plan = campaign.plan(lease=lease)
    reopened = world.open()
    assert reopened.load_plan(plan.to_json()) == plan
    getattr(world.fake, ending)(1000)
    with pytest.raises(StalePlan, match="ended or reached its pre-limit signal"):
        reopened.submit(plan)
    assert world.fake.count("sh") == 0


def test_steps_of_k_slots_ask_for_k_times_the_resources(world: World) -> None:
    campaign, (lease, *_) = leased(world)
    plan = campaign.plan(lease=lease, slots=2, only=["task-0"])
    (item,) = plan.allocations
    assert (item.slots, item.cpus, item.memory_mib) == (2, 4, 2048)
    assert b"--cpus-per-task=4 --mem=2048M" in item.script


def test_a_budget_reaches_the_step_and_holds_a_task_with_no_room(world: World) -> None:
    budget = Continuation(budget="00:30:00")
    campaign, lease, steps = running(world, 1, continuation=budget)
    assert b"SERVATUS_BUDGET=1800 " in world.fake.files[f"{LOGS}/{lease}.d/{steps['task-0']}.run"]
    world.fake.begin_step(steps["task-0"])
    world.clock.advance(minutes=30)
    world.fake.end_step(steps["task-0"], 75)
    assert campaign.plan(lease=lease, retry=Retry.FAILED, continuation=budget).decision.held == {
        "task-0": Hold.SPENT
    }
    larger = Continuation(budget="01:00:00")
    plan = campaign.plan(lease=lease, retry=["task-0"], continuation=larger)
    assert plan.decision.spent == {"task-0": 30 * MINUTE}
    assert b"previous=1800\n" in plan.allocations[0].script


def test_logs_of_a_step_are_its_slot_0_output(world: World) -> None:
    campaign, _, steps = running(world, 1)
    step = steps["task-0"]
    world.fake.write_file(f"{LOGS}/{step}-1000-0.out", b"step output")
    assert campaign.read_log(task="task-0").content == b"step output"
    assert campaign.read_log(allocation=step).content == b"step output"


# --- retry in the same lease -----------------------------------------------------------------


def test_a_task_retries_in_its_lease_as_soon_as_its_step_ended(world: World) -> None:
    campaign, lease, steps = running(world, 1)
    world.fake.begin_step(steps["task-0"])
    assert campaign.plan(lease=lease, retry=Retry.FAILED).decision.held == {"task-0": Hold.ACTIVE}
    with pytest.raises(PlanRefused, match="Slurm still lists earlier work"):
        campaign.plan(lease=lease, retry=["task-0"])
    world.fake.end_step(steps["task-0"], 1)
    assert world.fake.job(1000).state == "RUNNING"
    again = ordered(campaign, lease, retry=["task-0"])
    assert world.fake.poll(1000) == (again["task-0"],)
    assert outcome(campaign) == (Execution.QUEUED, None, None)


# --- cancel ----------------------------------------------------------------------------------


@pytest.mark.parametrize("started", [True, False], ids=["running", "pending"])
def test_cancel_before_the_lease_took_the_order_withdraws_it(world: World, started: bool) -> None:
    campaign, (lease, *_) = leased(world, 1)
    if started:
        world.fake.start(1000)
    steps = ordered(campaign, lease)
    step = steps["task-0"]
    assert campaign.cancel(tasks=["task-0"]) == (Receipt(step, 1000, ("task-0",)),)
    assert outcome(campaign) == (Execution.STOPPED, None, StopReason.NEVER_STARTED)
    assert campaign.plan(lease=lease, retry=["task-0"]).decision.selected == ("task-0",)
    if not started:
        world.fake.start(1000)
    assert world.fake.poll(1000) == ()
    assert marker(step, "void") in world.fake.files


def test_cancel_of_a_step_waiting_for_room_stops_it_unstarted(world: World) -> None:
    campaign, _, steps = running(world, 1)
    step = steps["task-0"]
    assert campaign.cancel(allocations=[step]) == (Receipt(step, 1000, ("task-0",)),)
    assert outcome(campaign) == (Execution.QUEUED, None, None)
    assert world.fake.honor_stops(1000) == (step,)
    assert outcome(campaign) == (Execution.STOPPED, 143, StopReason.NEVER_STARTED)


@pytest.mark.parametrize(
    ("code", "expected"),
    [(143, (Execution.STOPPED, 143, None)), (0, (Execution.SUCCEEDED, 0, None))],
)
def test_cancel_of_a_running_step_asks_the_lease_to_stop_it(
    world: World, code: int, expected: tuple[object, ...]
) -> None:
    campaign, _, steps = running(world, 2)
    world.fake.begin_step(steps["task-0"])
    world.fake.begin_step(steps["task-1"])
    campaign.cancel(tasks=["task-0"])
    assert world.fake.honor_stops(1000, code) == (steps["task-0"],)
    assert outcome(campaign) == expected
    assert outcome(campaign, "task-1") == (Execution.RUNNING, None, None)
    assert world.fake.job(1000).state == "RUNNING"


def test_cancel_of_an_ended_step_asks_nothing(world: World) -> None:
    campaign, _, steps = running(world, 1)
    world.fake.end_step(steps["task-0"])
    calls = world.fake.count("sh")
    assert campaign.cancel(tasks=["task-0"]) == ()
    assert world.fake.count("sh") == calls


def test_a_failed_withdrawal_names_the_step_and_what_was_cancelled(world: World) -> None:
    campaign, _, steps = running(world, 2)
    world.fake.fail_next("sh")
    with pytest.raises(Unavailable, match=r"cancel failed for \w{24} \(.*injected") as raised:
        campaign.cancel(tasks=["task-0", "task-1"])
    message = str(raised.value)
    failed, done = message[len("cancel failed for ") :][:24], message.rsplit(": ", 1)[1]
    assert {failed, done} == set(steps.values())


def test_status_names_each_attempt_kind(world: World) -> None:
    campaign, lease, steps = running(world, 1)
    kinds = {attempt.allocation_id: attempt.kind for attempt in campaign.status().attempts}
    assert kinds == {lease: Kind.LEASE, steps["task-0"]: Kind.STEP}


def test_cancel_of_a_lease_stops_its_steps_and_its_later_links(world: World) -> None:
    campaign, (first, second) = leased(world, links=2)
    world.fake.start(1000)
    steps = ordered(campaign, first)
    world.fake.poll(1000)
    world.fake.begin_step(steps["task-0"])
    assert campaign.cancel(allocations=[first]) == (
        Receipt(second, 1001, ()),
        Receipt(first, 1000, ()),
    )
    assert outcome(campaign) == (Execution.ENDING, None, None)
    world.fake.purge(1000)
    world.fake.purge(1001)
    assert outcome(campaign) == (Execution.STOPPED, None, None)
    assert outcome(campaign, "task-1") == (Execution.STOPPED, None, StopReason.NEVER_STARTED)


# --- the end of a lease ----------------------------------------------------------------------


def test_a_step_that_exits_nonzero_after_the_pre_limit_signal_stopped_at_the_limit(
    world: World,
) -> None:
    campaign, _, steps = running(world, 1)
    world.fake.begin_step(steps["task-0"])
    world.fake.signal_limit(1000)
    world.fake.end_step(steps["task-0"], 143)
    assert outcome(campaign) == (Execution.STOPPED, 143, StopReason.TIME_LIMIT)


@pytest.mark.parametrize(
    ("ending", "expected"),
    [("interrupt", (Execution.STOPPED, None, None)), ("lose", (Execution.LOST, None, None))],
)
def test_a_step_running_when_its_lease_ends(
    world: World, ending: str, expected: tuple[object, ...]
) -> None:
    campaign, _, steps = running(world, 1)
    world.fake.begin_step(steps["task-0"])
    world.clock.advance(minutes=20)
    getattr(world.fake, ending)(1000)
    assert outcome(campaign) == (Execution.ENDING, None, None)
    world.fake.purge(1000)
    assert outcome(campaign) == expected


# --- unresolved steps ------------------------------------------------------------------------


def interrupted_order(world: World, fault: str) -> tuple[Campaign, str, str]:
    """A running lease and one step whose order reached ``fault``; the lease and step ids."""
    campaign, (lease, *_) = leased(world, 1)
    world.fake.start(1000)
    plan = campaign.plan(lease=lease)
    getattr(world.fake, fault)("sh")
    with pytest.raises(SubmissionInterrupted, match="unresolved"):
        campaign.submit(plan)
    return campaign, lease, plan.allocations[0].allocation_id


def test_an_order_that_never_arrived_is_not_submitted_at_once(world: World) -> None:
    campaign, lease, step = interrupted_order(world, "fail_next")
    result = campaign.reconcile()
    assert result.receipts == () and result.waiting == ()
    assert [item.allocation_id for item in result.not_submitted] == [step]
    world.fake.run((*ORDER, f"{LOGS}/{lease}.d", step), stdin=b"previous=0\n")
    assert world.fake.poll(1000) == ()
    assert task(campaign).execution is AcceptanceState.NOT_SUBMITTED
    assert campaign.plan(lease=lease).decision.selected == ("task-0",)


def test_an_order_taken_before_its_reply_was_lost_is_accepted(world: World) -> None:
    campaign, _, step = interrupted_order(world, "lose_next_reply")
    assert world.fake.poll(1000) == (step,)
    assert campaign.reconcile().receipts == (Receipt(step, 1000, ("task-0",)),)
    assert outcome(campaign) == (Execution.QUEUED, None, None)


def test_a_not_submitted_step_that_started_is_an_evidence_conflict(world: World) -> None:
    campaign, _, step = interrupted_order(world, "lose_next_reply")
    campaign.reconcile()
    world.fake.write_file(marker(step, "start"))
    with pytest.raises(EvidenceConflict, match="recorded as not submitted, yet it ran"):
        campaign.status()


def test_a_withdrawn_step_that_started_is_an_evidence_conflict(world: World) -> None:
    campaign, (lease, *_) = leased(world, 1)
    world.fake.start(1000)
    step = ordered(campaign, lease)["task-0"]
    campaign.cancel(tasks=["task-0"])
    world.fake.write_file(marker(step, "start"))
    with pytest.raises(EvidenceConflict, match="was withdrawn, yet it ran"):
        campaign.status()


# --- interleavings ---------------------------------------------------------------------------


class Interleaving:
    """Two Tasks and a chain of two 2-slot leases (jobs 1000 and 1001), moved by operator and
    cluster actions that each pick among the moves possible right now."""

    def __init__(self, root: Path) -> None:
        self.world = make_world(root)
        self.campaign, (self.lease, _) = leased(self.world, links=2)
        self.taken: set[str] = set()

    def jobs(self, *states: str) -> list[int]:
        return [job.job for job in self.world.fake.jobs if job.in_queue and job.state in states]

    def eligible(self, job: int) -> bool:
        return job == 1000 or self.world.fake.job(1000).state not in ("PENDING", "RUNNING")

    def live_steps(self, *, begun: bool | None = None) -> list[str]:
        """Steps their lease took that have not ended, begun or not."""
        files = self.world.fake.files
        found: list[str] = []
        for attempt in self.campaign.status(scheduler=False).attempts:
            step, job = attempt.allocation_id, attempt.job
            if step not in self.taken or job is None or self.world.fake.job(job).state != "RUNNING":
                continue
            ended = any(path.startswith(marker(step, "0.", job)) for path in files)
            started = marker(step, "start", job) in files
            if not ended and begun in (None, started):
                found.append(step)
        return found

    def act(self, action: str, pick: int) -> bool:
        """Take ``action`` on its ``pick``-th possible target; whether one was possible."""
        fake, campaign = self.world.fake, self.campaign
        self.world.clock.advance(minutes=1)
        key = f"task-{pick % 2}"
        if action == "plan":
            retry = Retry.FAILED if pick < 2 else [key]
            campaign.submit(campaign.plan(lease=self.lease, retry=retry))
        elif action == "cancel":
            campaign.cancel(tasks=[key])
        elif action in ("begin", "end"):
            steps = self.live_steps(begun=False if action == "begin" else pick > 0 or None)
            if not steps:
                return False
            step = steps[pick % len(steps)]
            fake.begin_step(step) if action == "begin" else fake.end_step(step, pick % 3)
        else:
            states, control = _CONTROLS[action]
            jobs = [job for job in self.jobs(*states) if action != "start" or self.eligible(job)]
            if not jobs:
                return False
            taken = getattr(fake, control)(jobs[pick % len(jobs)])
            if action == "poll":
                self.taken.update(taken)
        return True

    def running_twice(self) -> list[str]:
        """Tasks with two steps that began and have not ended while their leases are listed;
        observing the campaign must not fail."""
        files = self.world.fake.files
        running: list[str] = []
        for attempt in self.campaign.status().attempts:
            step, job = attempt.allocation_id, attempt.job
            if attempt.within is None or job is None or not self.world.fake.job(job).in_queue:
                continue
            ended = any(path.startswith(marker(step, "0.", job)) for path in files)
            if marker(step, "start", job) in files and not ended:
                running.extend(attempt.task_keys)
        return sorted(key for key in set(running) if running.count(key) > 1)


_CONTROLS = {
    "start": (("PENDING",), "start"),
    "poll": (("RUNNING",), "poll"),
    "honor": (("RUNNING",), "honor_stops"),
    "limit": (("RUNNING",), "signal_limit"),
    "interrupt": (("RUNNING",), "interrupt"),
    "purge": (("CANCELLED",), "purge"),
}
_ACTIONS = (
    *("plan", "poll", "begin", "end") * 4,
    *("start", "honor", "purge") * 2,
    *("cancel", "limit", "interrupt"),
)


@settings(max_examples=25, deadline=None)
@given(st.randoms(use_true_random=False))
def test_no_task_ever_runs_twice_at_once_in_its_leases(random: Random) -> None:
    with tempfile.TemporaryDirectory() as root:
        machine = Interleaving(Path(root))
        moves = 0
        while moves < 30:
            with contextlib.suppress(Conflict, PlanRefused):
                if not machine.act(random.choice(_ACTIONS), random.randrange(6)):
                    continue
            moves += 1
            assert machine.running_twice() == []
