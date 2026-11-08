"""Budgeted continuation: chains of allocations planned, submitted, progressed, and extended
through ``FakeScheduler``, which emulates Slurm dependencies and the chain scripts' handoffs."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import pytest
from campaign_world import Probe, World, jobs
from support.builders import resources, target

from servatus.campaign import (
    Campaign,
    Completed,
    Continuation,
    Execution,
    Extension,
    Hold,
    Link,
    Plan,
    Profile,
    Retry,
    RunTime,
    StopReason,
    Totals,
)
from servatus.errors import (
    ConfigurationError,
    PlanRefused,
    StalePlan,
    SubmissionInterrupted,
    Unavailable,
)

BUDGET = Continuation(budget="03:30:00")
HOUR = timedelta(hours=1)


def chain_profile(**changes: Any) -> Profile:
    """Two-CPU Tasks with a 1-hour limit and a 10-minute pre-limit signal."""
    values: dict[str, Any] = {
        "container": None,
        "gpu_gres": None,
        "max_gpus_per_allocation": 0,
        "max_cpus_per_allocation": 8,
        "max_memory_mib_per_allocation": 4096,
        "max_time_limit": "1-00:00:00",
    }
    values.update(changes)
    limits = resources(
        cpus=2, memory_mib=1024, gpus=0, time_limit="01:00:00", signal_before_end="00:10:00"
    )
    return Profile("chain", target(**values), limits)


def submitted(world: World, count: int = 1, **options: Any) -> tuple[Campaign, Plan]:
    campaign = world.create(jobs(count))
    plan = campaign.plan(chain_profile(), continuation=BUDGET, **options)
    campaign.submit(plan)
    return campaign, plan


def time_limit(world: World, job: int, after: timedelta = HOUR) -> None:
    """The job runs ``after``, getting the pre-limit signal 10 minutes before Slurm stops it."""
    world.clock.advance(seconds=(after - timedelta(minutes=10)).total_seconds())
    world.fake.signal_limit(job)
    world.clock.advance(minutes=10)
    world.fake.interrupt(job, "TIMEOUT")


def states(world: World) -> list[str]:
    return [job.state for job in world.fake.jobs]


def purge_all(world: World) -> None:
    for job in world.fake.jobs:
        world.fake.purge(job.job)


def test_a_budget_plans_full_links_and_a_shorter_last_link(world: World) -> None:
    campaign = world.create(jobs(2))
    plan = campaign.plan(chain_profile(), continuation=BUDGET)
    links = [item.link for item in plan.allocations]
    first = plan.allocations[0].allocation_id
    budget = timedelta(hours=3, minutes=30)
    assert links == [Link(first, index, 4, budget) for index in range(1, 5)]
    assert [item.time_limit for item in plan.allocations] == [HOUR] * 3 + [timedelta(minutes=30)]
    assert [item.argv[-1] for item in plan.allocations] == ["--signal=B:USR1@600"] * 3 + [
        "--signal=USR1@600"
    ]
    assert all(item.task_keys == ("task-0", "task-1") for item in plan.allocations)
    assert dict(plan.decision.spent) == {"task-0": timedelta(0), "task-1": timedelta(0)}
    document = json.loads(plan.to_json())
    assert document["continuation"] == {"budget": "03:30:00", "max_links": None}
    assert document["spent"] == {"task-0": "00:00:00", "task-1": "00:00:00"}
    assert plan.warnings == () and plan.extensions == ()
    assert world.open().load_plan(plan.to_json()) == plan


def test_without_a_budget_max_links_plans_that_many_full_links(world: World) -> None:
    campaign = world.create(jobs(1))
    plan = campaign.plan(chain_profile(), continuation=Continuation(max_links=3))
    assert [(item.link.index, item.time_limit) for item in plan.allocations] == [
        (1, HOUR),
        (2, HOUR),
        (3, HOUR),
    ]
    assert b"SERVATUS_BUDGET" not in plan.allocations[0].script
    assert b"SERVATUS_LINKS=3" in plan.allocations[0].script


@pytest.mark.parametrize(
    ("continuation", "message"),
    [
        (Continuation(budget="10:00:00"), "needs resources.signal_before_end"),
        (Continuation(max_links=2), "needs resources.signal_before_end"),
    ],
)
def test_continuing_needs_the_pre_limit_signal_and_fails_before_contact(
    world: World, continuation: Continuation, message: str
) -> None:
    campaign = world.create(jobs(1))
    limits = resources(cpus=2, memory_mib=1024, gpus=0, time_limit="01:00:00")
    plain = Profile("plain", chain_profile().target, limits)
    with pytest.raises(ConfigurationError, match=message):
        campaign.plan(plain, continuation=continuation)
    with pytest.raises(ConfigurationError, match="must exceed signal_before_end"):
        campaign.plan(chain_profile(), continuation=Continuation(budget="00:11:00"))
    assert world.fake.calls == ()
    assert len(campaign.plan(plain, continuation=Continuation(max_links=1)).allocations) == 1


@pytest.mark.parametrize(
    ("budget", "max_links", "count", "warning"),
    [
        ("03:05:00", None, 3, "the last minutes of the budget of task-0 are dropped"),
        ("03:30:00", 2, 2, "2 links cannot spend the whole budget of task-0"),
        ("03:11:00", None, 3, "the last minutes of the budget of task-0 are dropped"),
        ("03:05:00", 3, 3, "3 links cannot spend the whole budget of task-0"),
        ("03:12:00", None, 4, None),
    ],
)
def test_what_a_budget_cannot_spend_is_a_warning(
    world: World, budget: str, max_links: int | None, count: int, warning: str | None
) -> None:
    campaign = world.create(jobs(1))
    continuation = Continuation(budget=budget, max_links=max_links)
    plan = campaign.plan(chain_profile(), continuation=continuation)
    assert len(plan.allocations) == count
    assert (warning is None) == (plan.warnings == ())
    assert warning is None or warning in plan.warnings[0]


def test_packed_tasks_with_different_histories_continue_for_their_own_links(
    world: World,
) -> None:
    campaign = world.create(jobs(2))
    campaign.submit(campaign.plan(chain_profile(), only=["task-0"]))
    world.fake.start(1000)
    world.clock.advance(minutes=50)
    world.fake.finish(1000)
    world.fake.purge(1000)
    plan = campaign.plan(chain_profile(), retry=["task-0"], continuation=BUDGET)
    assert plan.decision.spent["task-0"] == timedelta(minutes=50)
    assert plan.extensions == ("task-0",)
    assert plan.decision.extensions == {"task-0": Extension.SUCCEEDED}
    assert len(plan.allocations) == 4
    assert "different links (task-0 02:40:00 in 3, task-1 03:30:00 in 4)" in plan.warnings[0]
    third, fourth = plan.allocations[2].script, plan.allocations[3].script
    assert b'"$next.0.go"' not in third and b'"$next.1.go"' in third
    assert b"if handed 0;" not in fourth and b"if handed 1;" in fourth
    assert plan.allocations[2].time_limit == HOUR
    assert b"SERVATUS_RUN_TIME" in plan.allocations[0].script
    assert b"previous=3000\n" in plan.allocations[0].script


def test_a_chain_counts_every_link_against_max_allocations_per_submit(world: World) -> None:
    campaign = world.create(jobs(3))
    profile = chain_profile(max_allocations_per_submit=5, max_tasks_per_allocation=1)
    plan = campaign.plan(profile, continuation=Continuation(budget="02:00:00"))
    assert plan.decision.selected == ("task-0", "task-1")
    assert plan.decision.deferred == ("task-2",)
    assert len(plan.allocations) == 4
    tight = chain_profile(max_allocations_per_submit=1)
    assert campaign.plan(tight, continuation=BUDGET).decision.selected == ()


def test_chains_of_different_lengths_share_one_plan(world: World) -> None:
    campaign = world.create(jobs(2))
    campaign.submit(campaign.plan(chain_profile(), only=["task-0"]))
    world.fake.start(1000)
    world.clock.advance(minutes=50)
    world.fake.finish(1000)
    world.fake.purge(1000)
    profile = chain_profile(max_tasks_per_allocation=1)
    plan = campaign.plan(profile, retry=["task-0"], continuation=BUDGET)
    assert [(item.task_keys, item.link.index) for item in plan.allocations] == [
        (("task-0",), 1),
        (("task-1",), 1),
        (("task-0",), 2),
        (("task-1",), 2),
        (("task-0",), 3),
        (("task-1",), 3),
        (("task-1",), 4),
    ]
    assert plan.allocations[4].time_limit == timedelta(minutes=40)
    assert plan.warnings == ()
    assert plan.totals == Totals(12.34, 0.0, 6.17), "370 link minutes, rounded up"


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"max_links": 0}, "max_links must be an integer from 1 to 32"),
        ({"max_links": 33}, "max_links must be an integer from 1 to 32"),
        ({"max_links": True}, "max_links must be an integer from 1 to 32"),
        ({"budget": "soon"}, "budget: duration must use"),
        ({"budget": timedelta(0)}, "budget must be positive"),
    ],
)
def test_impossible_continuations_are_refused(value: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        Continuation(**value)


def test_a_budget_rounds_up_to_whole_minutes_and_plans_need_a_continuation(world: World) -> None:
    assert Continuation(budget="01:00:01").budget == timedelta(minutes=61)
    assert not Continuation().active and Continuation(max_links=1).active
    campaign = world.create(jobs(1))
    with pytest.raises(ConfigurationError, match="continuation must be a Continuation"):
        campaign.plan(chain_profile(), continuation="03:00:00")  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"continuation": {"budget": None, "max_links": None}}, "run times"),
        ({"extensions": {"task-0": "VALID"}}, "under a budget"),
        (
            {"continuation": {"budget": "01:00:00", "max_links": None}},
            "whose budget is already spent",
        ),
    ],
)
def test_inconsistent_chain_decisions_are_refused(
    world: World, changes: dict[str, object], message: str
) -> None:
    campaign = world.create(jobs(1))
    plan = campaign.plan(chain_profile(), continuation=Continuation(max_links=2))
    document = json.loads(plan.to_json()) | {"spent": {"task-0": "01:00:00"}} | changes
    with pytest.raises(ConfigurationError, match=message):
        campaign.load_plan(json.dumps(document).encode())


def test_tasks_packed_with_different_last_links_are_a_warning(world: World) -> None:
    campaign = world.create(jobs(2))
    campaign.submit(campaign.plan(chain_profile(), only=["task-0"]))
    world.fake.start(1000)
    world.clock.advance(minutes=30)
    world.fake.finish(1000, exit_code=1)
    world.fake.purge(1000)
    budget = Continuation(budget="03:00:00")
    plan = campaign.plan(chain_profile(), retry=["task-0"], continuation=budget)
    assert [item.time_limit for item in plan.allocations] == [HOUR] * 3
    (warning,) = plan.warnings
    assert "different links (task-0 02:30:00 in 3, task-1 03:00:00 in 3)" in warning


def test_a_tampered_continuation_breaks_the_digest(world: World) -> None:
    campaign = world.create(jobs(1))
    document = json.loads(campaign.plan(chain_profile(), continuation=BUDGET).to_json())
    document["continuation"]["budget"] = "04:00:00"
    with pytest.raises(ConfigurationError, match="digest does not match"):
        campaign.load_plan(json.dumps(document).encode())
    document["spent"] = {}
    with pytest.raises(ConfigurationError, match="run times do not match"):
        campaign.load_plan(json.dumps(document).encode())


def test_links_are_submitted_in_order_each_depending_on_every_earlier_link(
    world: World,
) -> None:
    campaign = world.create(jobs(5))
    plan = campaign.plan(chain_profile(max_tasks_per_allocation=3), continuation=BUDGET)
    result = campaign.submit(plan)
    assert [item.link.index for item in plan.allocations] == [1, 1, 2, 2, 3, 3, 4, 4]
    assert [receipt.job for receipt in result.receipts] == list(range(1000, 1008))
    assert [job.dependencies for job in world.fake.jobs] == [
        (),
        (),
        (1000,),
        (1001,),
        (1000, 1002),
        (1001, 1003),
        (1000, 1002, 1004),
        (1001, 1003, 1005),
    ]
    third = world.fake.jobs[4].argv
    assert third[-2:] == ("--dependency=afternotok:1000:1002", "--kill-on-invalid-dep=yes")
    assert not any(word.startswith("--dependency") for word in world.fake.jobs[0].argv)
    status = campaign.status()
    assert [attempt.link.index for attempt in status.attempts] == [1, 1, 2, 2, 3, 3, 4, 4]
    assert all(task.execution is Execution.QUEUED for task in status.tasks)
    assert [task.link and task.link.index for task in status.tasks] == [1] * 5


def test_a_lost_reply_on_a_link_leaves_it_unresolved_and_later_links_unattempted(
    world: World,
) -> None:
    campaign = world.create(jobs(1))
    plan = campaign.plan(chain_profile(), continuation=BUDGET)
    for hook in (None, None, "lose"):
        world.wire.on(
            "sbatch",
            lambda _argv, hook=hook: world.fake.lose_next_reply("sbatch") if hook else None,
        )
    with pytest.raises(SubmissionInterrupted) as raised:
        campaign.submit(plan)
    result = raised.value.result
    assert [receipt.job for receipt in result.receipts] == [1000, 1001]
    assert [item.allocation_id for item in result.unresolved] == [plan.allocations[2].allocation_id]
    assert [item.allocation_id for item in result.unattempted] == [
        plan.allocations[3].allocation_id
    ]
    assert [receipt.job for receipt in campaign.reconcile().receipts] == [1002]
    with pytest.raises(StalePlan):
        campaign.submit(plan)
    assert len(world.fake.jobs) == 3


def test_a_link_whose_earlier_link_was_purged_is_refused_and_reconciled_away(
    world: World,
) -> None:
    campaign = world.create(jobs(1))
    plan = campaign.plan(chain_profile(), continuation=Continuation(max_links=2))

    def purge_first(_argv: tuple[str, ...]) -> None:
        world.fake.finish(1000)
        world.fake.purge(1000)

    world.wire.on("sbatch", lambda _argv: None)
    world.wire.on("sbatch", purge_first)
    with pytest.raises(SubmissionInterrupted, match="acceptance is unresolved"):
        campaign.submit(plan)
    assert len(world.fake.jobs) == 1
    world.clock.advance(minutes=16)
    assert len(campaign.reconcile().not_submitted) == 1


def test_a_changed_run_time_makes_the_plan_stale_before_any_intent(world: World) -> None:
    campaign = world.create(jobs(1))
    campaign.submit(campaign.plan(chain_profile()))
    world.fake.start(1000)
    world.clock.advance(minutes=20)
    world.fake.finish(1000)
    world.fake.purge(1000)
    plan = campaign.plan(chain_profile(), retry=["task-0"], continuation=BUDGET)
    assert plan.decision.spent == {"task-0": timedelta(minutes=20)}
    marker = next(path for path in world.fake.files if path.endswith(".0.0.end"))
    world.fake.remove_file(marker)
    revision = world.revision()
    with pytest.raises(StalePlan, match="measured run time changed"):
        campaign.submit(plan)
    assert world.revision() == revision


def test_validate_checks_later_links_with_kill_on_invalid_dep(world: World) -> None:
    campaign = world.create(jobs(1))
    plan = campaign.plan(chain_profile(), continuation=BUDGET)
    checks = campaign.validate(plan)
    assert [check.time_limit for check in checks] == [HOUR, HOUR, timedelta(minutes=30)]
    tested = [call for call in world.fake.calls if "--test-only" in call]
    assert ["--kill-on-invalid-dep=yes" in call for call in tested] == [False, True, True]
    assert not any(word.startswith("--dependency") for call in tested for word in call)
    assert world.fake.jobs == ()


def test_a_time_limit_hands_the_task_to_the_next_link_and_exit_0_ends_the_chain(
    world: World,
) -> None:
    campaign, plan = submitted(world)
    world.fake.start(1000)
    time_limit(world, 1000)
    root = chain_profile().target.log_root
    assert world.fake.files[f"{root}/{plan.allocations[1].allocation_id}.0.go"] == b"3600\n"
    world.fake.start(1001)
    world.clock.advance(minutes=5)
    (task,) = campaign.status().tasks
    assert (task.execution, task.link) == (Execution.RUNNING, plan.allocations[1].link)
    assert task.run_time == RunTime(timedelta(minutes=65), False)
    world.fake.finish(1001)
    assert states(world) == ["TIMEOUT", "COMPLETED", "CANCELLED", "CANCELLED"]
    with pytest.raises(ConfigurationError, match="is not pending"):
        world.fake.start(1002)
    purge_all(world)
    status = campaign.status()
    (task,) = status.tasks
    assert (task.execution, task.exit_code, task.current_allocation_id) == (
        Execution.SUCCEEDED,
        0,
        plan.allocations[1].allocation_id,
    )
    assert task.run_time == RunTime(timedelta(minutes=65), True)
    assert status.quiescent


@pytest.mark.parametrize(
    ("signalled", "code", "execution", "detail", "continued"),
    [
        (True, 138, Execution.STOPPED, StopReason.TIME_LIMIT, True),
        (True, 75, Execution.STOPPED, StopReason.YIELDED, True),
        (False, 75, Execution.STOPPED, StopReason.YIELDED, True),
        (False, 1, Execution.FAILED, None, False),
        (True, 0, Execution.SUCCEEDED, None, False),
    ],
)
def test_how_a_step_ends_decides_whether_its_chain_continues(
    world: World,
    signalled: bool,
    code: int,
    execution: Execution,
    detail: StopReason | None,
    continued: bool,
) -> None:
    campaign, _ = submitted(world)
    world.fake.start(1000)
    world.clock.advance(minutes=30)
    if signalled:
        world.fake.signal_limit(1000)
    world.fake.finish(1000, exit_code=code)
    world.fake.purge(1000)
    assert states(world)[0] == ("FAILED" if continued else "COMPLETED")
    assert states(world)[1:] == ["PENDING" if continued else "CANCELLED"] * 3
    if not continued:
        purge_all(world)
    status = campaign.status()
    (task,) = status.tasks
    first, second = status.attempts[:2]
    assert first.observation is not None
    assert first.observation.execution(0) == (execution, code, detail)
    assert task.current_allocation_id == (second if continued else first).allocation_id
    assert task.execution is (Execution.QUEUED if continued else execution)


def test_a_cancel_without_the_signal_hands_nothing_on(world: World) -> None:
    campaign, plan = submitted(world)
    world.fake.start(1000)
    world.clock.advance(minutes=20)
    world.fake.interrupt(1000)
    world.fake.start(1001)
    assert states(world) == ["CANCELLED", "COMPLETED", "CANCELLED", "CANCELLED"]
    purge_all(world)
    (task,) = campaign.status().tasks
    assert (task.execution, task.detail) == (Execution.STOPPED, None)
    assert task.current_allocation_id == plan.allocations[0].allocation_id
    assert task.run_time == RunTime(timedelta(minutes=20), True)


def test_a_lost_link_counts_its_whole_limit_and_waits_for_an_explicit_retry(
    world: World,
) -> None:
    campaign, plan = submitted(world)
    world.fake.start(1000)
    time_limit(world, 1000)
    world.fake.start(1001)
    world.fake.lose(1001)
    world.fake.start(1002)
    assert states(world) == ["TIMEOUT", "NODE_FAIL", "COMPLETED", "CANCELLED"]
    purge_all(world)
    (task,) = campaign.status().tasks
    assert (task.execution, task.current_allocation_id) == (
        Execution.LOST,
        plan.allocations[1].allocation_id,
    )
    assert task.run_time == RunTime(2 * HOUR, False)
    assert campaign.plan(chain_profile(), retry=Retry.FAILED).decision.selected == ()
    retry = campaign.plan(chain_profile(), retry=["task-0"], continuation=BUDGET)
    assert retry.decision.spent == {"task-0": 2 * HOUR}
    assert [item.time_limit for item in retry.allocations] == [HOUR, timedelta(minutes=30)]


def test_every_listed_link_holds_the_task_active(world: World) -> None:
    campaign, _ = submitted(world)
    world.fake.start(1000)
    time_limit(world, 1000)
    world.fake.purge(1000)
    plan = campaign.plan(chain_profile(), retry=Retry.FAILED, continuation=BUDGET)
    assert plan.decision.held == {"task-0": Hold.ACTIVE}
    with pytest.raises(PlanRefused, match="Slurm still lists earlier work"):
        campaign.plan(chain_profile(), retry=["task-0"], continuation=BUDGET)


def test_logs_follow_the_task_to_its_current_link(world: World) -> None:
    campaign, plan = submitted(world)
    world.fake.start(1000)
    time_limit(world, 1000)
    world.fake.start(1001)
    root = chain_profile().target.log_root
    world.fake.write_file(f"{root}/{plan.allocations[1].allocation_id}-1001-0.out", b"link 2")
    assert campaign.read_log(task="task-0").content == b"link 2"


def test_cancel_stops_a_chain_newest_link_first(world: World) -> None:
    campaign, plan = submitted(world)
    world.fake.start(1000)
    receipts = campaign.cancel(tasks=["task-0"])
    assert [receipt.job for receipt in receipts] == [1003, 1002, 1001, 1000]
    cancels = [call[-1] for call in world.fake.calls if call[0].endswith("scancel")]
    names = [f"--name=servatus-{item.allocation_id}" for item in reversed(plan.allocations)]
    assert cancels == names
    assert states(world) == ["CANCELLED"] * 4


def test_a_failed_cancel_leaves_the_earlier_links_of_its_chain_alone(world: World) -> None:
    campaign, plan = submitted(world)
    world.fake.start(1000)
    world.wire.on("scancel", lambda _argv: None)
    world.wire.on("scancel", lambda _argv: Completed(1, b"", b"scancel: error\n"))
    with pytest.raises(Unavailable, match="not attempted: ") as raised:
        campaign.cancel(tasks=["task-0"])
    first, second = plan.allocations[0].allocation_id, plan.allocations[1].allocation_id
    assert raised.value.args[0].endswith(f"not attempted: {second}, {first}")
    assert states(world) == ["RUNNING", "PENDING", "PENDING", "CANCELLED"]


def test_cancelling_a_chain_with_an_unresolved_link_needs_reconcile_first(world: World) -> None:
    campaign = world.create(jobs(1))
    plan = campaign.plan(chain_profile(), continuation=Continuation(max_links=2))
    world.wire.on("sbatch", lambda _argv: None)
    world.wire.on("sbatch", lambda _argv: world.fake.lose_next_reply("sbatch"))
    with pytest.raises(SubmissionInterrupted):
        campaign.submit(plan)
    unresolved = plan.allocations[1].allocation_id
    with pytest.raises(Exception, match=f"allocation {unresolved} is unresolved"):
        campaign.cancel(tasks=["task-0"])
    campaign.reconcile()
    assert len(campaign.cancel(tasks=["task-0"])) == 2


def test_cancelling_one_link_cuts_the_chain_there(world: World) -> None:
    campaign, plan = submitted(world)
    receipts = campaign.cancel(allocations=[plan.allocations[2].allocation_id])
    assert [receipt.job for receipt in receipts] == [1003, 1002]
    assert states(world) == ["PENDING", "PENDING", "CANCELLED", "CANCELLED"]


def finished(world: World, probe: Probe | None = None) -> Campaign:
    """task-0 ran 40 minutes and exited 0; Slurm forgot it."""
    campaign = world.create(jobs(1), probe=probe)
    campaign.submit(campaign.plan(chain_profile()))
    world.fake.start(1000)
    world.clock.advance(minutes=40)
    world.fake.finish(1000)
    world.fake.purge(1000)
    return campaign


def test_a_larger_budget_extends_a_succeeded_task_once(world: World) -> None:
    campaign = finished(world)
    extend = Continuation(budget="01:30:00")
    plan = campaign.plan(chain_profile(), retry=["task-0"], continuation=extend)
    assert plan.extensions == ("task-0",) and plan.retry == ("task-0",)
    assert [item.time_limit for item in plan.allocations] == [timedelta(minutes=50)]
    script = plan.allocations[0].script
    assert b"SERVATUS_BUDGET=5400" in script and b"previous=2400\n" in script
    campaign.submit(plan)
    world.fake.start(1001)
    world.clock.advance(minutes=50)
    world.fake.finish(1001)
    world.fake.purge(1001)
    (task,) = campaign.status().tasks
    assert task.run_time == RunTime(timedelta(minutes=90), True)
    held = campaign.plan(chain_profile(), retry=Retry.FAILED, continuation=extend)
    assert held.decision.held == {"task-0": Hold.SPENT}
    with pytest.raises(PlanRefused, match="budget is already spent"):
        campaign.plan(chain_profile(), retry=["task-0"], continuation=extend)


def test_a_bulk_retry_of_incomplete_work_under_a_budget_extends_and_submits(world: World) -> None:
    probe = Probe()
    campaign = finished(world, probe)
    plan = campaign.plan(chain_profile(), retry=Retry.INCOMPLETE, continuation=BUDGET)
    assert plan.decision.extensions == {"task-0": Extension.SUCCEEDED}
    assert len(campaign.submit(plan).receipts) == 3


def test_the_last_link_reads_a_signal_death_as_a_failure_like_a_plain_allocation(
    world: World,
) -> None:
    campaign, _ = submitted(world)
    for job in (1000, 1001, 1002):
        world.fake.start(job)
        time_limit(world, job)
    world.fake.start(1003)
    world.clock.advance(minutes=20)
    world.fake.signal_limit(1003)
    world.fake.finish(1003, exit_code=138)
    assert "limit" not in "".join(path for path in world.fake.files if "1003" in path)
    purge_all(world)
    (task,) = campaign.status().tasks
    assert (task.execution, task.exit_code, task.detail) == (Execution.FAILED, 138, None)
    assert task.run_time == RunTime(timedelta(hours=3, minutes=20), True)


def test_a_retry_of_failed_work_under_a_budget_holds_spent_tasks(world: World) -> None:
    campaign = world.create(jobs(2))
    campaign.submit(campaign.plan(chain_profile(max_tasks_per_allocation=1)))
    for job, minutes in ((1000, 50), (1001, 10)):
        world.fake.start(job)
        world.clock.advance(minutes=minutes)
        world.fake.finish(job, exit_code=1)
        world.fake.purge(job)
    plan = campaign.plan(
        chain_profile(), retry=Retry.FAILED, continuation=Continuation(budget="00:55:00")
    )
    assert plan.decision.selected == ("task-1",)
    assert plan.decision.held == {"task-0": Hold.SPENT}
    assert plan.extensions == ()


def test_a_valid_result_is_extended_only_by_name_under_a_budget(world: World) -> None:
    probe = Probe()
    campaign = finished(world, probe)
    probe.valid.add("task-0")
    with pytest.raises(PlanRefused, match="valid results cannot be retried"):
        campaign.plan(chain_profile(), retry=["task-0"])
    plain = campaign.plan(chain_profile(), retry=Retry.FAILED, continuation=BUDGET)
    assert plain.decision.held == {"task-0": Hold.VALID}
    plan = campaign.plan(chain_profile(), retry=["task-0"], continuation=BUDGET)
    assert plan.decision.extensions == {"task-0": Extension.VALID}
    probe.valid.clear()
    with pytest.raises(StalePlan, match="what the planned extensions extend changed"):
        campaign.submit(plan)


def test_a_failed_submission_of_the_first_link_records_nothing_more(world: World) -> None:
    campaign = world.create(jobs(1))
    plan = campaign.plan(chain_profile(), continuation=BUDGET)
    world.fake.fail_next("sbatch", Completed(1, b"", b"sbatch: error: MaxSubmitJobs\n"))
    with pytest.raises(SubmissionInterrupted, match="acceptance is unresolved") as raised:
        campaign.submit(plan)
    assert len(raised.value.result.unattempted) == 3
    assert isinstance(raised.value.__cause__, Unavailable)
