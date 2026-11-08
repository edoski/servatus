from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import timedelta

import pytest
from core_helpers import NOW, allocation_id, evidence, intend, roster, scenarios, seen, submit
from hypothesis import given, settings
from hypothesis import strategies as st

from servatus.campaign._evidence import Execution, Observation, Observed, StepEnd, StopReason
from servatus.campaign._state import AcceptanceState, Link, State, history, record_outcome, seal
from servatus.campaign._status import (
    STATUS_FORMAT,
    ResultState,
    RunTime,
    Status,
    classify_results,
    current,
    project,
    run_time,
)
from servatus.errors import ConfigurationError

E = Execution


def view(
    state: State,
    observations: Mapping[str, Observation] | None = None,
    results: dict[str, ResultState] | None = None,
    *,
    strays: Mapping[str, int] | None = None,
    scheduler_observed: bool = True,
) -> Status:
    return project(
        state,
        evidence(observations, strays),
        results or {},
        scheduler_observed=scheduler_observed,
        observed_at=NOW,
    )


def test_fresh_roster_is_unsubmitted_and_trivially_quiescent() -> None:
    status = view(roster(2))
    assert [task.key for task in status.tasks] == ["task-0", "task-1"]
    assert all(
        task.current_allocation_id is None
        and task.execution is AcceptanceState.NOT_SUBMITTED
        and task.exit_code is None
        and task.detail is None
        and task.result is ResultState.UNOBSERVED
        for task in status.tasks
    )
    assert status.attempts == () and status.warnings == () and status.quiescent
    assert not view(roster(2), scheduler_observed=False).quiescent
    assert (status.campaign_id, status.revision, status.sealed) == ("c" * 32, 0, True)
    assert status.observed_at == NOW and status.scheduler_observed


def test_attempts_project_their_durable_facts_and_observation() -> None:
    state, first = submit(roster(2), ["task-0", "task-1"])
    state, second = submit(state, ["task-1"])
    observation = seen(None, 0, 1)
    status = view(state, {first: observation})
    one, two = status.attempts
    assert (one.allocation_id, one.task_keys, one.retry) == (first, ("task-0", "task-1"), ())
    assert (two.allocation_id, two.retry, two.profile_label) == (second, ("task-1",), "test")
    assert one.acceptance is AcceptanceState.ACCEPTED and one.job == 101
    assert one.intent_at == NOW and one.observation == observation
    assert two.job == 102 and two.observation is None


@pytest.mark.parametrize(
    ("observation", "execution", "exit_code", "detail"),
    [
        (seen("PENDING"), E.QUEUED, None, None),
        (seen("PENDING", 0, started=False), E.QUEUED, None, None),
        (seen("REQUEUE_HOLD"), E.QUEUED, None, None),
        (seen("SOMETHING_NEW"), E.QUEUED, None, None),
        (seen("RUNNING"), E.RUNNING, None, None),
        (seen("COMPLETING"), E.RUNNING, None, None),
        (seen("SUSPENDED"), E.RUNNING, None, None),
        (seen("FAILED"), E.ENDING, None, None),
        (seen("CANCELLED", stopped=True), E.ENDING, None, None),
        (seen("RUNNING", 0), E.SUCCEEDED, 0, None),
        (seen("COMPLETED", 0), E.SUCCEEDED, 0, None),
        (seen("CANCELLED", 3, stopped=True), E.FAILED, 3, None),
        (seen(None, 0), E.SUCCEEDED, 0, None),
        (seen(None, 255, stopped=True), E.FAILED, 255, None),
        (seen(None, stopped=True), E.STOPPED, None, None),
        (seen(None, started=False), E.STOPPED, None, "never started"),
        (seen(None, started=False, stopped=True), E.STOPPED, None, "never started"),
        (seen(None), E.LOST, None, None),
    ],
)
def test_every_execution_reaches_the_task_through_its_observation(
    observation: Observation, execution: Execution, exit_code: int | None, detail: str | None
) -> None:
    state, identity = submit(roster(1), ["task-0"])
    (task,) = view(state, {identity: observation}).tasks
    assert (task.execution, task.exit_code, task.detail) == (execution, exit_code, detail)
    assert task.current_allocation_id == identity


def test_each_slot_reads_its_own_end_marker() -> None:
    state, identity = submit(roster(3), ["task-0", "task-1", "task-2"])
    tasks = view(state, {identity: seen(None, 0, 2, None, stopped=True)}).tasks
    assert [(task.execution, task.exit_code) for task in tasks] == [
        (E.SUCCEEDED, 0),
        (E.FAILED, 2),
        (E.STOPPED, None),
    ]
    tasks = view(state, {identity: seen("RUNNING", 0, None, None)}).tasks
    assert [task.execution for task in tasks] == [E.SUCCEEDED, E.RUNNING, E.RUNNING]


def test_latest_accepted_attempt_owns_execution_but_quiescence_uses_all() -> None:
    state, first = submit(roster(1), ["task-0"])
    state, second = submit(state, ["task-0"])
    status = view(state, {first: seen("RUNNING"), second: seen(None, 0)})
    (task,) = status.tasks
    assert task.current_allocation_id == second
    assert (task.execution, task.exit_code) == (E.SUCCEEDED, 0)
    assert not status.quiescent
    assert view(state, {first: seen(None, 1), second: seen(None, 0)}).quiescent


def test_unresolved_acceptance_dominates_until_resolved() -> None:
    state, first = submit(roster(1), ["task-0"])
    pending, second = intend(state, ["task-0"])
    observations = {first: seen(None, 0)}
    status = view(pending, observations)
    (task,) = status.tasks
    assert task.execution is AcceptanceState.UNRESOLVED and task.current_allocation_id == second
    assert task.exit_code is None
    assert not status.quiescent
    assert status.attempts[1].acceptance is AcceptanceState.UNRESOLVED
    after = view(record_outcome(pending, second, None), observations)
    assert [attempt.acceptance for attempt in after.attempts] == [
        AcceptanceState.ACCEPTED,
        AcceptanceState.NOT_SUBMITTED,
    ]
    (task,) = after.tasks
    assert task.current_allocation_id == first
    assert task.execution is E.SUCCEEDED and after.quiescent


def test_not_submitted_only_history_leaves_the_task_unsubmitted() -> None:
    state, identity = submit(roster(1), ["task-0"], accepted=False)
    status = view(state, {identity: seen("RUNNING")})
    assert status.tasks[0].current_allocation_id is None
    assert status.tasks[0].execution is AcceptanceState.NOT_SUBMITTED
    assert status.attempts[0].observation is None, "only accepted attempts carry evidence"
    assert status.quiescent


def test_unobserved_accepted_work_shows_acceptance_and_is_not_quiescent() -> None:
    state, _ = submit(roster(1), ["task-0"])
    status = view(state)
    assert status.tasks[0].execution is AcceptanceState.ACCEPTED
    assert status.tasks[0].current_allocation_id
    assert not status.quiescent
    assert not view(state, scheduler_observed=False).quiescent


@pytest.mark.parametrize(
    ("observation", "quiescent"),
    [
        (seen("PENDING"), False),
        (seen("RUNNING"), False),
        (seen("COMPLETED", 0), False),
        (seen("CANCELLED", stopped=True), False),
        (seen(None, 0), True),
        (seen(None, 1), True),
        (seen(None, stopped=True), True),
        (seen(None, started=False), True),
        (seen(None), True),
    ],
)
def test_quiescence_waits_until_slurm_purges_every_accepted_job(
    observation: Observation, quiescent: bool
) -> None:
    state, identity = submit(roster(1), ["task-0"])
    assert view(state, {identity: observation}).quiescent is quiescent
    assert not view(state, {identity: observation}, scheduler_observed=False).quiescent


def test_a_stray_blocks_quiescence_and_is_named_in_a_warning() -> None:
    state, accepted = submit(roster(2), ["task-0"])
    state, stray = submit(state, ["task-1"], accepted=False)
    observations = {accepted: seen(None, 0)}
    status = view(state, observations, strays={stray: 4242})
    assert not status.quiescent
    (warning,) = status.warnings
    assert stray in warning and "job 4242" in warning
    assert status.tasks[1].current_allocation_id is None
    gone = view(state, observations)
    assert gone.quiescent and gone.warnings == ()


def test_results_ready_requires_a_sealed_roster_of_valid_results() -> None:
    valid = {"task-0": ResultState.VALID, "task-1": ResultState.VALID}
    assert view(roster(2), results=valid).results_ready
    growing = roster(2, appendable=True)
    assert not view(growing, results=valid).results_ready
    assert view(seal(growing), results=valid).results_ready
    partial = {"task-0": ResultState.VALID, "task-1": ResultState.MISSING}
    assert not view(roster(2), results=partial).results_ready
    assert not view(roster(2), results={"task-0": ResultState.VALID}).results_ready
    statuses = view(roster(2), results={"task-0": ResultState.MISSING}).tasks
    assert [task.result for task in statuses] == [ResultState.MISSING, ResultState.UNOBSERVED]


def test_classify_results_from_one_probe_answer() -> None:
    assert classify_results(["a", "b"], {"b"}) == {
        "a": ResultState.MISSING,
        "b": ResultState.VALID,
    }
    assert classify_results([], ()) == {}
    with pytest.raises(ConfigurationError, match="not asked about: 'c'"):
        classify_results(["a"], ["a", "c"])
    for answer in ("a", b"a", None, 3):
        with pytest.raises(ConfigurationError, match="collection of Task keys"):
            classify_results(["a"], answer)  # pyright: ignore[reportArgumentType]


def test_counts_partition_tasks_by_result_and_execution() -> None:
    state, first = submit(roster(9), ["task-0", "task-1", "task-2", "task-3"])
    state, second = submit(state, ["task-4"])
    state, _ = submit(state, ["task-5"])
    state, _ = submit(state, ["task-6"], accepted=False)
    state, _ = intend(state, ["task-7"])
    observations = {first: seen(None, 0, 1, None, None), second: seen("PENDING")}
    results = {"task-0": ResultState.VALID, "task-1": ResultState.MISSING}
    counts = view(state, observations, results).counts()
    assert dict(counts) == {
        "tasks": 9,
        "valid": 1,
        "missing": 1,
        "unobserved": 7,
        "not_submitted": 2,
        "unresolved": 1,
        "accepted": 1,
        "queued": 1,
        "running": 0,
        "ending": 0,
        "succeeded": 1,
        "failed": 1,
        "stopped": 0,
        "lost": 2,
    }
    running = view(state, {first: seen("RUNNING", None, None, 0, None, stopped=True)}).counts()
    assert (running["running"], running["succeeded"]) == (3, 1)
    ending = view(state, {first: seen("CANCELLED", None, None, 0, 0, stopped=True)}).counts()
    assert (ending["ending"], ending["succeeded"]) == (2, 2)
    stopped = view(state, {first: seen(None, None, None, 0, 0, stopped=True)}).counts()
    assert (stopped["stopped"], stopped["succeeded"]) == (2, 2)


def test_json_is_versioned_canonical_and_round_trips() -> None:
    state, first = submit(roster(3), ["task-0", "task-2"])
    state, stray = submit(state, ["task-1"], accepted=False)
    state, pending = intend(state, ["task-1"])
    observations = {first: seen("COMPLETED", 1, None)}
    status = view(state, observations, {"task-0": ResultState.MISSING}, strays={stray: 77})
    data = status.to_json()
    value = json.loads(data)
    assert value["format"] == STATUS_FORMAT == "servatus.status/3"
    assert data == json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    assert value["counts"] == dict(status.counts())
    assert value["observed_at"] == "2030-01-01T00:00:00+00:00"
    assert value["results_ready"] is False and value["quiescent"] is False
    assert value["scheduler_observed"] is True
    assert value["warnings"] == list(status.warnings) and len(status.warnings) == 1
    assert value["tasks"][0] == {
        "key": "task-0",
        "result": "MISSING",
        "current_allocation_id": first,
        "execution": "FAILED",
        "exit_code": 1,
        "detail": None,
        "run_time": {"total": "00:10:00", "complete": True},
        "link": {"chain": first, "index": 1, "count": 1, "budget": None},
    }
    assert value["tasks"][2]["run_time"] == {"total": "00:00:00", "complete": False}
    attempt = value["attempts"][0]
    assert attempt["job"] == 101
    assert attempt["link"] == value["tasks"][0]["link"]
    assert attempt["observation"] == {
        "slurm_state": "COMPLETED",
        "started_at": "2030-01-01T00:00:00+00:00",
        "stopped_at": None,
        "limit_at": None,
        "ends": [{"exit_code": 1, "at": "2030-01-01T00:10:00+00:00"}, None],
        "handed": [True, True],
    }
    assert attempt["intent_at"] == "2030-01-01T00:00:00+00:00"
    assert value["attempts"][1]["observation"] is None
    assert value["attempts"][2]["allocation_id"] == pending
    assert value["attempts"][2]["acceptance"] == "UNRESOLVED"
    assert set(value) == {
        "format",
        "campaign_id",
        "revision",
        "sealed",
        "observed_at",
        "scheduler_observed",
        "tasks",
        "attempts",
        "results_ready",
        "quiescent",
        "warnings",
        "counts",
    }
    assert "args" not in data.decode() and "stdin" not in data.decode()
    assert value["tasks"][1]["execution"] == "UNRESOLVED"


# --- properties ------------------------------------------------------------------------------


@settings(max_examples=400, deadline=None)
@given(scenarios(every_accepted_observed=False), st.booleans())
def test_projection_invariants(
    scenario: tuple[State, Observed, dict[str, ResultState]], scheduler: bool
) -> None:
    state, observed, results = scenario
    status = project(state, observed, results, scheduler_observed=scheduler, observed_at=NOW)
    counts = status.counts()
    assert counts["tasks"] == len(state.tasks)
    assert counts["valid"] + counts["missing"] + counts["unobserved"] == counts["tasks"]
    execution = sum(
        value
        for name, value in counts.items()
        if name not in {"tasks", "valid", "missing", "unobserved"}
    )
    assert execution == counts["tasks"]
    for durable, projected in zip(state.attempts, status.attempts, strict=True):
        earlier = {
            key
            for other in state.attempts
            if other.job is not None
            and other.outcome_revision is not None
            and other.outcome_revision < durable.intent_revision
            for key in other.task_keys
        }
        assert projected.retry == tuple(key for key in durable.task_keys if key in earlier)
    owners = {attempt.allocation_id: attempt for attempt in status.attempts}
    for task in status.tasks:
        assert (task.exit_code is not None) <= (
            task.execution in (E.SUCCEEDED, E.FAILED, E.STOPPED)
        )
        assert task.detail is None or task.execution is E.STOPPED
        assert scheduler or task.run_time is None
        if task.current_allocation_id is None:
            assert task.execution is AcceptanceState.NOT_SUBMITTED
            continue
        owner = owners[task.current_allocation_id]
        assert task.key in owner.task_keys
        assert owner.acceptance is not AcceptanceState.NOT_SUBMITTED
        unresolved = task.execution is AcceptanceState.UNRESOLVED
        assert unresolved == (owner.acceptance is AcceptanceState.UNRESOLVED)
        later = status.attempts[status.attempts.index(owner) + 1 :]
        assert not any(
            task.key in attempt.task_keys
            and attempt.acceptance is not AcceptanceState.NOT_SUBMITTED
            for attempt in later
        )
        assert isinstance(task.execution, Execution) == (owner.observation is not None)
    assert len(status.warnings) == len(observed.strays)
    if status.quiescent:
        assert scheduler and not observed.strays
        assert AcceptanceState.UNRESOLVED not in {task.execution for task in status.tasks}
        assert all(
            attempt.observation is not None and not attempt.observation.live
            for attempt in status.attempts
            if attempt.acceptance is AcceptanceState.ACCEPTED
        )
    assert status.results_ready == (
        state.sealed and all(results.get(task.key) is ResultState.VALID for task in state.tasks)
    )


# --- chains ----------------------------------------------------------------------------------

BUDGET = timedelta(hours=3)


def chain(links: int = 3) -> tuple[State, list[str]]:
    """A ``links``-link budgeted chain of task-0 and task-1, every link accepted."""
    state, first = submit(
        roster(2), ["task-0", "task-1"], link=Link(allocation_id(1), 1, links, BUDGET)
    )
    identities = [first]
    for index in range(2, links + 1):
        state, identity = submit(
            state, ["task-0", "task-1"], link=Link(first, index, links, BUDGET)
        )
        identities.append(identity)
    return state, identities


def handed(*slots: bool, state: str | None = None, code: int | None = None) -> Observation:
    """A gated link that started at ``NOW`` with ``slots`` handed; a returned step ended 10
    minutes later."""
    end = None if code is None else StepEnd(code, NOW + timedelta(minutes=10))
    return Observation(state, NOW, None, None, (end,) * len(slots), slots)


def test_the_current_allocation_is_the_latest_link_that_was_handed_the_task() -> None:
    state, (first, second, third) = chain()
    past = history(state)
    observed = evidence(
        {
            first: seen(None, 75, 75),
            second: handed(True, False, code=75),
            third: handed(False, False),
        }
    )
    assert current(past["task-0"], observed) == (state.attempt(second), 0)
    assert current(past["task-1"], observed) == (state.attempt(first), 1)
    assert current(past["task-0"], evidence()) == (state.attempt(first), 0)


def test_a_task_never_accepted_has_no_current_allocation_or_run_time() -> None:
    state, _ = intend(roster(1), ["task-0"])
    past = history(state)["task-0"]
    assert current(past, evidence()) is None
    assert run_time(past, evidence(), NOW) == RunTime(timedelta(0), True)


def test_run_time_sums_every_link_and_counts_unseen_links_as_unknown() -> None:
    state, (first, second, third) = chain()
    past = history(state)["task-0"]
    observed = evidence(
        {
            first: seen(None, 75, 75),
            second: handed(True, True, code=75),
            third: handed(True, True, state="RUNNING"),
        }
    )
    later = NOW + timedelta(minutes=25, microseconds=999_999)
    assert run_time(past, observed, later) == RunTime(timedelta(minutes=45), False)
    assert run_time(past, evidence({first: seen(None, 75, 75)}), later) is None


def test_status_shows_the_chain_link_its_run_time_and_why_it_stopped() -> None:
    state, (first, second, third) = chain()
    observed = {
        first: seen(None, 75, 75),
        second: handed(True, True, code=75),
        third: handed(True, False, code=75),
    }
    status = view(state, observed)
    zero, one = status.tasks
    assert (zero.current_allocation_id, zero.link) == (third, state.attempt(third).link)
    assert (zero.execution, zero.exit_code, zero.detail) == (E.STOPPED, 75, StopReason.BUDGET_SPENT)
    assert zero.run_time == RunTime(timedelta(minutes=30), True)
    assert (one.current_allocation_id, one.detail) == (second, StopReason.YIELDED)
    assert one.run_time == RunTime(timedelta(minutes=20), True)
    assert [attempt.link.index for attempt in status.attempts] == [1, 2, 3]
    offline = view(state, scheduler_observed=False)
    assert offline.tasks[0].current_allocation_id == first
    assert offline.tasks[0].run_time is None
