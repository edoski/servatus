from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone

import pytest
from core_helpers import NOW, intend, observed, roster, step, submit
from hypothesis import given, settings
from hypothesis import strategies as st

from servatus.campaign._evidence import AllocationState, JobRef, Observation, SchedulerEvidence
from servatus.campaign._state import AcceptanceState, State, seal
from servatus.campaign._status import (
    STATUS_FORMAT,
    ResultState,
    Status,
    classify_results,
    project,
    task_execution,
)
from servatus.errors import ConfigurationError

S = AllocationState


def view(
    state: State,
    observations: dict[str, Observation] | None = None,
    results: dict[str, ResultState] | None = None,
    *,
    scheduler_observed: bool = True,
) -> Status:
    return project(
        state,
        observations or {},
        results or {},
        scheduler_observed=scheduler_observed,
        observed_at=NOW,
    )


def test_fresh_roster_is_unsubmitted_and_trivially_quiescent() -> None:
    status = view(roster(2))
    assert [task.key for task in status.tasks] == ["task-0", "task-1"]
    assert all(
        task.current_allocation_id is None
        and task.execution is None
        and task.exit_code is None
        and not task.unresolved
        and task.result is ResultState.UNOBSERVED
        for task in status.tasks
    )
    assert status.attempts == () and status.quiescent
    assert not view(roster(2), scheduler_observed=False).quiescent
    assert (status.campaign_id, status.revision, status.sealed) == ("c" * 32, 0, True)
    assert status.observed_at == NOW and status.scheduler_observed


def test_attempts_project_their_durable_facts_and_evidence() -> None:
    state, first = submit(roster(2), ["task-0", "task-1"])
    state, second = submit(state, ["task-1"], ack=["task-1"])
    evidence = observed(S.FAILED, exit_code="1:0", steps=(step(S.FAILED, "1:0"), None))
    status = view(state, {first: evidence})
    one, two = status.attempts
    assert (one.allocation_id, one.task_keys, one.retry, one.duplicate_risk) == (
        first,
        ("task-0", "task-1"),
        (),
        (),
    )
    assert (two.retry, two.duplicate_risk, two.profile_label) == (("task-1",), ("task-1",), "test")
    assert one.acceptance is AcceptanceState.ACCEPTED and one.job == JobRef(101)
    assert one.intent_at == NOW and one.scheduler == evidence.allocation
    assert one.steps == evidence.steps
    assert two.scheduler is None and two.steps == () and two.allocation_id == second


def test_latest_accepted_attempt_owns_execution_but_quiescence_uses_all() -> None:
    state, first = submit(roster(1), ["task-0"])
    state, second = submit(state, ["task-0"])
    running = {first: observed(S.RUNNING), second: observed(S.SUCCEEDED, exit_code="0:0")}
    status = view(state, running)
    (task,) = status.tasks
    assert task.current_allocation_id == second
    assert (task.execution, task.exit_code) == (S.SUCCEEDED, "0:0")
    assert not status.quiescent
    done = {first: observed(S.FAILED, exit_code="1:0"), second: observed(S.SUCCEEDED)}
    assert view(state, done).quiescent


def test_unresolved_acceptance_dominates_until_resolved() -> None:
    state, first = submit(roster(1), ["task-0"])
    pending, second = intend(state, ["task-0"])
    evidence = {first: observed(S.SUCCEEDED)}
    status = view(pending, evidence)
    (task,) = status.tasks
    assert task.unresolved and task.current_allocation_id == second
    assert task.execution is None and task.exit_code is None
    assert not status.quiescent
    assert status.attempts[1].acceptance is AcceptanceState.UNRESOLVED
    from servatus.campaign._state import record_outcome

    resolved = record_outcome(pending, second, None)
    after = view(resolved, evidence)
    assert [attempt.acceptance for attempt in after.attempts] == [
        AcceptanceState.ACCEPTED,
        AcceptanceState.NOT_SUBMITTED,
    ]
    (task,) = after.tasks
    assert not task.unresolved and task.current_allocation_id == first
    assert task.execution is S.SUCCEEDED and after.quiescent


def test_not_submitted_only_history_leaves_the_task_unsubmitted() -> None:
    state, identity = submit(roster(1), ["task-0"], job=None)
    status = view(state, {identity: observed(S.FAILED)})
    assert status.tasks[0].current_allocation_id is None
    assert status.attempts[0].scheduler is None, "only accepted attempts carry evidence"
    assert status.quiescent


def test_step_evidence_overrides_allocation_evidence_per_task() -> None:
    state, identity = submit(roster(3), ["task-0", "task-1", "task-2"])
    evidence = observed(
        S.FAILED,
        exit_code="1:0",
        steps=(step(S.SUCCEEDED, "0:0"), step(S.FAILED, "2:0")),  # task-2 has no step entry
    )
    tasks = view(state, {identity: evidence}).tasks
    assert [(task.execution, task.exit_code) for task in tasks] == [
        (S.SUCCEEDED, "0:0"),
        (S.FAILED, "2:0"),
        (S.UNKNOWN, None),
    ]
    assert task_execution(evidence, 0, 3) == (S.SUCCEEDED, "0:0")
    assert task_execution(evidence, 9, 1) == (S.FAILED, "1:0")


@pytest.mark.parametrize("state", [S.FAILED, S.CANCELLED])
def test_a_failed_packed_allocation_without_the_task_step_is_unknown(state: S) -> None:
    """The Task may have finished successfully before a sibling failed the allocation."""
    packed, identity = submit(roster(2), ["task-0", "task-1"])
    for steps in ((), (None, None)):
        evidence = observed(state, exit_code="1:0", steps=steps)
        tasks = view(packed, {identity: evidence}).tasks
        assert [(task.execution, task.exit_code) for task in tasks] == [(S.UNKNOWN, None)] * 2
    single, alone = submit(roster(1), ["task-0"])
    (task,) = view(single, {alone: observed(state, exit_code="1:0")}).tasks
    assert (task.execution, task.exit_code) == (state, "1:0")
    succeeded = view(packed, {identity: observed(S.SUCCEEDED, exit_code="0:0")}).tasks
    assert [task.execution for task in succeeded] == [S.SUCCEEDED] * 2


def test_unknown_or_contradictory_allocations_ignore_step_evidence() -> None:
    state, identity = submit(roster(1), ["task-0"])
    steps = (step(S.SUCCEEDED, "0:0"),)
    contradictory = Observation(
        SchedulerEvidence(S.SUCCEEDED, exit_code="0:0", retained=True, problem="contradiction"),
        steps,
    )
    for evidence in (observed(S.UNKNOWN, steps=steps), contradictory):
        (task,) = view(state, {identity: evidence}).tasks
        assert (task.execution, task.exit_code) == (S.UNKNOWN, None)


@pytest.mark.parametrize("state", [S.QUEUED, S.RUNNING])
def test_exit_codes_are_shown_only_for_terminal_executions(state: S) -> None:
    packed, identity = submit(roster(2), ["task-0", "task-1"])
    evidence = observed(state, exit_code="0:0", steps=(step(state, "0:0"), None))
    tasks = view(packed, {identity: evidence}).tasks
    assert [(task.execution, task.exit_code) for task in tasks] == [(state, None)] * 2


def test_unobserved_accepted_work_has_no_execution_and_is_not_quiescent() -> None:
    state, _ = submit(roster(1), ["task-0"])
    status = view(state)
    assert status.tasks[0].execution is None and status.tasks[0].current_allocation_id
    assert not status.quiescent
    assert not view(state, scheduler_observed=False).quiescent


@pytest.mark.parametrize(
    ("evidence", "quiescent"),
    [
        (observed(S.SUCCEEDED), True),
        (observed(S.FAILED), True),
        (observed(S.CANCELLED), True),
        (observed(S.SUCCEEDED, retained=True), False),
        (observed(S.UNKNOWN), False),
        (observed(S.QUEUED), False),
        (observed(S.RUNNING), False),
    ],
)
def test_quiescence_requires_terminal_unretained_allocations(
    evidence: Observation, quiescent: bool
) -> None:
    state, identity = submit(roster(1), ["task-0"])
    assert view(state, {identity: evidence}).quiescent is quiescent
    assert not view(state, {identity: evidence}, scheduler_observed=False).quiescent


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


def test_observed_at_must_be_utc() -> None:
    for at in (datetime(2030, 1, 1), datetime(2030, 1, 1, tzinfo=timezone(timedelta(hours=2)))):  # noqa: DTZ001
        with pytest.raises(ConfigurationError, match="aware UTC"):
            project(roster(1), {}, {}, scheduler_observed=False, observed_at=at)


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
    state, first = submit(roster(6), ["task-0", "task-1"])
    state, _ = submit(state, ["task-2"])
    state, _ = submit(state, ["task-3"], job=None)
    state, _ = intend(state, ["task-4"])
    evidence = {first: observed(S.FAILED, steps=(step(S.SUCCEEDED), step(S.FAILED)))}
    results = {"task-0": ResultState.VALID, "task-1": ResultState.MISSING}
    counts = view(state, evidence, results).counts()
    assert dict(counts) == {
        "tasks": 6,
        "valid": 1,
        "missing": 1,
        "unobserved": 4,
        "unsubmitted": 2,
        "unresolved": 1,
        "accepted": 1,
        "queued": 0,
        "running": 0,
        "succeeded": 1,
        "failed": 1,
        "cancelled": 0,
        "unknown": 0,
    }


def test_json_is_versioned_canonical_and_self_contained() -> None:
    state, first = submit(roster(2), ["task-0"])
    state, second = intend(state, ["task-1"])
    evidence = {first: observed(S.FAILED, exit_code="1:0", retained=True, steps=(step(S.FAILED),))}
    status = view(state, evidence, {"task-0": ResultState.MISSING})
    data = status.to_json()
    value = json.loads(data)
    assert value["format"] == STATUS_FORMAT == "servatus.status/1"
    assert data == json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    assert value["counts"] == dict(status.counts())
    assert value["observed_at"] == "2030-01-01T00:00:00+00:00"
    assert value["results_ready"] is False and value["quiescent"] is False
    assert value["scheduler_observed"] is True
    task = value["tasks"][0]
    assert task == {
        "key": "task-0",
        "result": "MISSING",
        "current_allocation_id": first,
        "execution": "FAILED",
        "exit_code": None,
        "unresolved": False,
    }
    attempt = value["attempts"][0]
    assert attempt["job"] == {"job_id": 101, "cluster": None}
    assert attempt["scheduler"]["retained"] is True
    assert attempt["scheduler"]["exit_code"] == "1:0"
    assert attempt["steps"] == [{"state": "FAILED", "raw_state": "FAILED", "exit_code": None}]
    assert attempt["intent_at"] == "2030-01-01T00:00:00+00:00"
    assert value["attempts"][1]["allocation_id"] == second
    assert value["attempts"][1]["scheduler"] is None
    assert value["attempts"][1]["acceptance"] == "UNRESOLVED"
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
        "counts",
    }
    assert "args" not in data.decode() and "stdin" not in data.decode()


# --- properties ------------------------------------------------------------------------------

STATES = st.sampled_from(list(AllocationState))


@st.composite
def scenarios(draw: st.DrawFn) -> tuple[State, dict[str, Observation], dict[str, ResultState]]:
    state = roster(3)
    for _ in range(draw(st.integers(0, 5))):
        busy = {
            key
            for attempt in state.attempts
            if attempt.acceptance is AcceptanceState.UNRESOLVED
            for key in attempt.task_keys
        }
        free = [task.key for task in state.tasks if task.key not in busy]
        if not free:
            break
        keys = draw(st.lists(st.sampled_from(free), min_size=1, max_size=3, unique=True))
        ordered = [key for key in free if key in keys]
        outcome = draw(st.sampled_from(["accept", "reject", "pending"]))
        if outcome == "pending":
            state, _ = intend(state, ordered)
        else:
            state, _ = submit(state, ordered, job=0 if outcome == "accept" else None)
    observations: dict[str, Observation] = {}
    for attempt in state.attempts:
        if draw(st.booleans()):
            steps = draw(st.lists(st.none() | STATES.map(step), max_size=3))
            observations[attempt.allocation_id] = observed(
                draw(STATES), retained=draw(st.booleans()), steps=steps
            )
    results = {
        task.key: draw(st.sampled_from(list(ResultState)))
        for task in state.tasks
        if draw(st.booleans())
    }
    return state, observations, results


@settings(max_examples=400, deadline=None)
@given(scenarios(), st.booleans())
def test_projection_invariants(
    scenario: tuple[State, dict[str, Observation], dict[str, ResultState]], scheduler: bool
) -> None:
    state, observations, results = scenario
    status = view(state, observations, results, scheduler_observed=scheduler)
    counts = status.counts()
    assert counts["tasks"] == len(state.tasks)
    assert counts["valid"] + counts["missing"] + counts["unobserved"] == counts["tasks"]
    execution = sum(
        value
        for name, value in counts.items()
        if name not in {"tasks", "valid", "missing", "unobserved"}
    )
    assert execution == counts["tasks"]
    assert status.to_json() == status.to_json()
    owners = {attempt.allocation_id: attempt for attempt in status.attempts}
    for task in status.tasks:
        if task.current_allocation_id is None:
            assert task.execution is None and not task.unresolved
            continue
        owner = owners[task.current_allocation_id]
        assert task.key in owner.task_keys
        assert owner.acceptance is not AcceptanceState.NOT_SUBMITTED
        assert task.unresolved == (owner.acceptance is AcceptanceState.UNRESOLVED)
        later = status.attempts[status.attempts.index(owner) + 1 :]
        assert not any(
            task.key in attempt.task_keys
            and attempt.acceptance is not AcceptanceState.NOT_SUBMITTED
            for attempt in later
        )
        if task.execution is not None:
            assert owner.scheduler is not None
    if status.quiescent:
        assert scheduler and not any(task.unresolved for task in status.tasks)
        assert all(
            attempt.scheduler is not None
            and attempt.scheduler.state.terminal
            and not attempt.scheduler.retained
            for attempt in status.attempts
            if attempt.acceptance is AcceptanceState.ACCEPTED
        )
    assert status.results_ready == (
        state.sealed and all(results.get(task.key) is ResultState.VALID for task in state.tasks)
    )


def test_projection_is_pure_and_utc_only() -> None:
    state, identity = submit(roster(1), ["task-0"])
    evidence = {identity: observed(S.RUNNING)}
    first = view(state, evidence)
    assert view(state, evidence) == first
    assert first.observed_at.tzinfo is UTC
