from __future__ import annotations

from collections.abc import Collection

import pytest
from core_helpers import NOW, intend, observed, roster, step, submit
from hypothesis import given, settings
from hypothesis import strategies as st
from support.builders import profile

from servatus.campaign._evidence import AllocationState, Observation
from servatus.campaign._policy import Hold, Retry, Selection, decide, observation_scope
from servatus.campaign._state import AcceptanceState, State, record_intent
from servatus.campaign._status import ResultState
from servatus.errors import ConfigurationError, PlanRefused

S = AllocationState
R = ResultState
KEY = "task-0"


def choose(
    state: State,
    observations: dict[str, Observation] | None = None,
    results: dict[str, ResultState] | None = None,
    *,
    retry: Collection[str] | Retry = (),
    duplicate_risk: Collection[str] = (),
    only: Collection[str] | None = None,
) -> Selection:
    return decide(
        state,
        observations or {},
        results or {},
        retry=retry,
        duplicate_risk=duplicate_risk,
        only=only,
    )


def once(evidence: Observation | None) -> tuple[State, dict[str, Observation]]:
    state, identity = submit(roster(1), [KEY])
    return state, {} if evidence is None else {identity: evidence}


def outcome(selection: Selection) -> str | Hold:
    """``selected``, ``retry`` (selected resubmission), or the hold reason for ``KEY``."""
    if KEY in selection.selected:
        return "retry" if KEY in selection.retry else "selected"
    return selection.held[KEY]


# --- fresh work ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (None, "selected"),
        (R.MISSING, "selected"),
        (R.UNOBSERVED, "selected"),
        (R.VALID, Hold.VALID),
    ],
)
@pytest.mark.parametrize("retry", [(), Retry.FAILED, Retry.INCOMPLETE, ("task-1",)])
def test_fresh_tasks_are_selected_unless_valid(
    result: ResultState | None, expected: str | Hold, retry: Collection[str] | Retry
) -> None:
    state = roster(2)
    state, _ = submit(state, ["task-1"])
    results = {} if result is None else {KEY: result}
    assert outcome(choose(state, results=results, retry=retry)) == expected


def test_not_submitted_history_is_still_fresh() -> None:
    state, _ = submit(roster(1), [KEY], job=None)
    assert outcome(choose(state)) == "selected"
    with pytest.raises(PlanRefused, match="retry needs an earlier accepted attempt"):
        choose(state, retry=[KEY])


def test_only_limits_the_plan_and_overrides_retry() -> None:
    state, _ = submit(roster(3), ["task-1"])
    selection = choose(state, only=["task-0", "task-1"], retry=["task-1", "task-2"])
    assert selection.selected == ("task-0",)
    assert dict(selection.held) == {"task-1": Hold.UNOBSERVABLE, "task-2": Hold.NOT_REQUESTED}
    assert choose(state, only=()).held == {
        "task-0": Hold.NOT_REQUESTED,
        "task-1": Hold.NOT_REQUESTED,
        "task-2": Hold.NOT_REQUESTED,
    }


# --- explicit retry --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        (observed(S.FAILED), "retry"),
        (observed(S.CANCELLED), "retry"),
        (observed(S.SUCCEEDED), "retry"),
        (observed(S.QUEUED), Hold.ACTIVE),
        (observed(S.RUNNING), Hold.ACTIVE),
        (observed(S.SUCCEEDED, retained=True), Hold.ACTIVE),
        (observed(S.UNKNOWN, retained=True), Hold.ACTIVE),
        (observed(S.FAILED, steps=(step(S.RUNNING),)), Hold.ACTIVE),
        (None, Hold.UNOBSERVABLE),
    ],
)
def test_explicit_retry_of_accepted_work(
    evidence: Observation | None, expected: str | Hold
) -> None:
    state, observations = once(evidence)
    selection = choose(state, observations, retry=[KEY])
    assert outcome(selection) == expected
    if expected == "retry":
        assert selection.retry == (KEY,) and selection.duplicate_risk == ()


def test_accepted_work_is_submitted_unless_retry_is_requested() -> None:
    for evidence in (None, observed(S.RUNNING), observed(S.FAILED), observed(S.UNKNOWN)):
        state, observations = once(evidence)
        assert outcome(choose(state, observations)) == Hold.SUBMITTED
        assert observation_scope(state) == ()


@pytest.mark.parametrize(
    "evidence",
    [observed(S.UNKNOWN), observed(S.FAILED, steps=(step(S.UNKNOWN),))],
)
def test_unknown_evidence_needs_acknowledgement(evidence: Observation) -> None:
    state, observations = once(evidence)
    with pytest.raises(PlanRefused, match="UNKNOWN scheduler evidence") as refused:
        choose(state, observations, retry=[KEY])
    assert refused.value.keys == (KEY,)
    selection = choose(state, observations, retry=[KEY], duplicate_risk=[KEY])
    assert selection.selected == selection.retry == selection.duplicate_risk == (KEY,)


def test_every_accepted_attempt_governs_retry_safety() -> None:
    state, first = submit(roster(1), [KEY])
    state, second = submit(state, [KEY])
    unknown_then_failed = {first: observed(S.UNKNOWN), second: observed(S.FAILED)}
    with pytest.raises(PlanRefused, match="duplicate-risk acknowledgement"):
        choose(state, unknown_then_failed, retry=[KEY])
    running_then_failed = {first: observed(S.RUNNING), second: observed(S.FAILED)}
    assert outcome(choose(state, running_then_failed, retry=[KEY])) == Hold.ACTIVE
    missing_first = {second: observed(S.FAILED)}
    assert outcome(choose(state, missing_first, retry=[KEY])) == Hold.UNOBSERVABLE
    assert observation_scope(state, [KEY]) == (first, second)


def test_acknowledgement_survives_improved_evidence() -> None:
    state, observations = once(observed(S.FAILED))
    selection = choose(state, observations, retry=[KEY], duplicate_risk=[KEY])
    assert selection.duplicate_risk == (KEY,)


def test_unresolved_work_is_held_even_when_retried() -> None:
    state, first = submit(roster(1), [KEY])
    state, _ = intend(state, [KEY])
    for retry in ([KEY], Retry.FAILED, Retry.INCOMPLETE, ()):
        assert outcome(choose(state, {first: observed(S.FAILED)}, retry=retry)) == Hold.UNRESOLVED
    pending, _ = intend(roster(1), [KEY])
    assert outcome(choose(pending, retry=[KEY])) == Hold.UNRESOLVED
    assert observation_scope(state, [KEY]) == () == observation_scope(state, Retry.FAILED)


def test_valid_results_are_never_resubmitted() -> None:
    state, observations = once(observed(S.FAILED))
    valid = {KEY: R.VALID}
    with pytest.raises(PlanRefused, match="valid results cannot be retried: 'task-0'"):
        choose(state, observations, valid, retry=[KEY], duplicate_risk=[KEY])
    for retry in (Retry.FAILED, Retry.INCOMPLETE, ()):
        assert outcome(choose(state, observations, valid, retry=retry)) == Hold.VALID


def test_all_offending_keys_are_refused_together() -> None:
    state = roster(5)
    state, first = submit(state, ["task-0", "task-1"])
    observations = {first: observed(S.UNKNOWN)}
    with pytest.raises(PlanRefused, match="^plan refused; ") as refused:
        choose(
            state,
            observations,
            {"task-0": R.VALID},
            retry=["task-0", "task-1", "task-2", "ghost"],
            duplicate_risk=["task-3", "phantom"],
            only=["task-0", "task-1", "task-2", "task-3", "nobody"],
        )
    error = refused.value
    assert error.keys == ("ghost", "nobody", "phantom", "task-0", "task-2", "task-1", "task-3")
    message = str(error)
    for reason in (
        "unknown Task keys: 'ghost', 'nobody', 'phantom'",
        "valid results cannot be retried: 'task-0'",
        "retry needs an earlier accepted attempt: 'task-2'",
        "UNKNOWN scheduler evidence needs a duplicate-risk acknowledgement: 'task-1'",
        "requires an explicitly retried Task key: 'task-3'",
    ):
        assert reason in message


def test_acknowledgement_requires_explicit_retry() -> None:
    state, observations = once(observed(S.UNKNOWN))
    for retry in ((), Retry.FAILED, Retry.INCOMPLETE):
        with pytest.raises(PlanRefused, match="explicitly retried") as refused:
            choose(state, observations, retry=retry, duplicate_risk=[KEY])
        assert refused.value.keys == (KEY,)


# --- bulk retry ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        (observed(S.FAILED), "retry"),
        (observed(S.CANCELLED), "retry"),
        (observed(S.SUCCEEDED), Hold.SUBMITTED),
        (observed(S.FAILED, steps=(step(S.SUCCEEDED),)), Hold.SUBMITTED),
        (observed(S.SUCCEEDED, steps=(step(S.FAILED),)), "retry"),
        (observed(S.SUCCEEDED, steps=(step(S.CANCELLED),)), "retry"),
        (observed(S.FAILED, steps=(None,)), "retry"),
        (observed(S.UNKNOWN), Hold.UNOBSERVABLE),
        (observed(S.FAILED, steps=(step(S.UNKNOWN),)), Hold.UNOBSERVABLE),
        (observed(S.RUNNING), Hold.ACTIVE),
        (observed(S.FAILED, retained=True), Hold.ACTIVE),
        (None, Hold.UNOBSERVABLE),
    ],
)
def test_retry_failed_uses_the_tasks_own_current_execution(
    evidence: Observation | None, expected: str | Hold
) -> None:
    state, observations = once(evidence)
    assert outcome(choose(state, observations, retry=Retry.FAILED)) == expected


def test_retry_failed_judges_the_latest_accepted_attempt() -> None:
    state, first = submit(roster(1), [KEY])
    state, second = submit(state, [KEY])
    evidence = {first: observed(S.FAILED), second: observed(S.SUCCEEDED)}
    assert outcome(choose(state, evidence, retry=Retry.FAILED)) == Hold.SUBMITTED
    evidence = {first: observed(S.SUCCEEDED), second: observed(S.CANCELLED)}
    assert outcome(choose(state, evidence, retry=Retry.FAILED)) == "retry"


@pytest.mark.parametrize(
    ("evidence", "result", "expected"),
    [
        (observed(S.SUCCEEDED), R.MISSING, "retry"),
        (observed(S.FAILED), R.MISSING, "retry"),
        (observed(S.SUCCEEDED), R.UNOBSERVED, Hold.UNOBSERVABLE),
        (observed(S.SUCCEEDED), None, Hold.UNOBSERVABLE),
        (observed(S.SUCCEEDED), R.VALID, Hold.VALID),
        (observed(S.RUNNING), R.MISSING, Hold.ACTIVE),
        (observed(S.UNKNOWN), R.MISSING, Hold.UNOBSERVABLE),
        (None, R.MISSING, Hold.UNOBSERVABLE),
    ],
)
def test_retry_incomplete_selects_terminal_missing_results(
    evidence: Observation | None, result: ResultState | None, expected: str | Hold
) -> None:
    state, observations = once(evidence)
    results = {} if result is None else {KEY: result}
    assert outcome(choose(state, observations, results, retry=Retry.INCOMPLETE)) == expected


def test_bulk_selection_mixes_fresh_and_resubmitted_work_in_roster_order() -> None:
    state = roster(4)
    state, first = submit(state, ["task-1", "task-2"])
    state, second = submit(state, ["task-3"])
    evidence = {
        first: observed(S.FAILED, steps=(step(S.FAILED), step(S.SUCCEEDED))),
        second: observed(S.CANCELLED),
    }
    selection = choose(state, evidence, retry=Retry.FAILED)
    assert selection.selected == ("task-0", "task-1", "task-3")
    assert selection.retry == ("task-1", "task-3")
    assert dict(selection.held) == {"task-2": Hold.SUBMITTED}
    with pytest.raises(TypeError, match="does not support item assignment"):
        selection.held["x"] = Hold.VALID  # pyright: ignore[reportIndexIssue]


# --- observation scope and inputs ------------------------------------------------------------


def test_observation_scope_covers_only_retryable_accepted_attempts() -> None:
    state = roster(4)
    state, first = submit(state, ["task-0", "task-1"])
    state, _ = submit(state, ["task-2"], job=None)
    state, third = submit(state, ["task-2"])
    state, fourth = submit(state, ["task-0"])
    state, _ = intend(state, ["task-3"])
    assert observation_scope(state) == ()
    assert observation_scope(state, ["task-3"]) == ()
    assert observation_scope(state, ["task-1"]) == (first,)
    assert observation_scope(state, ["task-0"]) == (first, fourth)
    assert observation_scope(state, ["task-2", "ghost"]) == (third,)
    assert observation_scope(state, Retry.FAILED) == (first, third, fourth)
    assert observation_scope(state, Retry.INCOMPLETE, only=["task-2"]) == (third,)
    assert observation_scope(state, ["task-0"], only=["task-1"]) == ()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"retry": "task-0"}, "retry must be a collection"),
        ({"retry": [1]}, "retry must contain only Task key strings"),
        ({"retry": None}, "retry must be a collection"),
        ({"duplicate_risk": "task-0"}, "duplicate_risk must be a collection"),
        ({"only": "task-0"}, "only must be a collection"),
        ({"only": [b"task-0"]}, "only must contain only"),
    ],
)
def test_malformed_requests_are_configuration_errors(
    options: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        choose(roster(1), **options)  # pyright: ignore[reportArgumentType]


def test_observation_scope_validates_request_types() -> None:
    with pytest.raises(ConfigurationError, match="retry must be a collection"):
        observation_scope(roster(1), "task-0")
    with pytest.raises(ConfigurationError, match="only must be a collection"):
        observation_scope(roster(1), (), "task-0")


# --- properties ------------------------------------------------------------------------------

KEYS = ("task-0", "task-1", "task-2")
STATES = st.sampled_from(list(AllocationState))


@st.composite
def scenarios(draw: st.DrawFn) -> tuple[State, dict[str, Observation], dict[str, ResultState]]:
    state = roster(len(KEYS))
    for _ in range(draw(st.integers(0, 5))):
        busy = {
            key
            for attempt in state.attempts
            if attempt.acceptance is AcceptanceState.UNRESOLVED
            for key in attempt.task_keys
        }
        free = [key for key in KEYS if key not in busy]
        if not free:
            break
        chosen = draw(st.lists(st.sampled_from(free), min_size=1, max_size=3, unique=True))
        keys = [key for key in free if key in chosen]
        kind = draw(st.sampled_from(["accept", "accept", "reject", "pending"]))
        if kind == "pending":
            state, _ = intend(state, keys)
        else:
            state, _ = submit(state, keys, job=0 if kind == "accept" else None)
    observations: dict[str, Observation] = {}
    for attempt in state.attempts:
        if draw(st.integers(0, 9)):
            steps = draw(st.lists(st.none() | STATES.map(step), max_size=3))
            observations[attempt.allocation_id] = observed(
                draw(STATES), retained=draw(st.booleans()), steps=steps
            )
    results = {key: draw(st.sampled_from(list(ResultState))) for key in KEYS if draw(st.booleans())}
    return state, observations, results


subsets = st.sets(st.sampled_from(KEYS)).map(lambda chosen: tuple(k for k in KEYS if k in chosen))
requests = st.one_of(subsets, st.sampled_from(list(Retry)))
onlys = st.none() | subsets


def accepted_history(state: State, key: str) -> list[tuple[str, int]]:
    return [
        (attempt.allocation_id, attempt.task_keys.index(key))
        for attempt in state.attempts
        if attempt.acceptance is AcceptanceState.ACCEPTED and key in attempt.task_keys
    ]


def unresolved(state: State, key: str) -> bool:
    return any(
        attempt.acceptance is AcceptanceState.UNRESOLVED and key in attempt.task_keys
        for attempt in state.attempts
    )


def task_states(observation: Observation, slot: int) -> tuple[AllocationState, AllocationState]:
    own = observation.steps[slot] if slot < len(observation.steps) else None
    return observation.allocation.state, observation.allocation.state if own is None else own.state


def expected_refusals(
    state: State,
    observations: dict[str, Observation],
    results: dict[str, ResultState],
    retry: tuple[str, ...] | Retry,
    ack: tuple[str, ...],
    only: tuple[str, ...] | None,
) -> set[str]:
    explicit = None if isinstance(retry, Retry) else set(retry)
    refused = {key for key in ack if explicit is None or key not in explicit}
    for key in explicit or ():
        if only is not None and key not in only:
            continue
        if results.get(key) is ResultState.VALID:
            refused.add(key)
            continue
        if unresolved(state, key):
            continue
        history = accepted_history(state, key)
        if not history:
            refused.add(key)
            continue
        evidence = [(observations.get(identity), slot) for identity, slot in history]
        if any(
            item is not None
            and (item.allocation.retained or any(value.active for value in task_states(item, slot)))
            for item, slot in evidence
        ):
            continue
        if any(item is None for item, _ in evidence):
            continue
        unknown = any(
            S.UNKNOWN in task_states(item, slot) for item, slot in evidence if item is not None
        )
        if unknown and key not in ack:
            refused.add(key)
    return refused


@settings(max_examples=2000, deadline=None)
@given(scenarios(), requests, subsets, onlys)
def test_policy_safety_properties(
    scenario: tuple[State, dict[str, Observation], dict[str, ResultState]],
    retry: tuple[str, ...] | Retry,
    ack: tuple[str, ...],
    only: tuple[str, ...] | None,
) -> None:
    state, observations, results = scenario
    expected = expected_refusals(state, observations, results, retry, ack, only)
    try:
        selection = choose(state, observations, results, retry=retry, duplicate_risk=ack, only=only)
    except PlanRefused as refusal:
        assert set(refusal.keys) == expected and expected
        assert len(refusal.keys) == len(set(refusal.keys))
        return
    assert not expected
    if isinstance(retry, Retry):
        assert not ack, "bulk selectors never take acknowledgements"
    # Partition, roster order.
    assert set(selection.selected).isdisjoint(selection.held)
    assert set(selection.selected) | set(selection.held) == set(KEYS)
    assert selection.selected == tuple(key for key in KEYS if key in selection.selected)
    assert selection.retry == tuple(k for k in selection.selected if accepted_history(state, k))
    assert set(selection.duplicate_risk) <= set(selection.retry) & set(ack)
    for key in selection.selected:
        assert only is None or key in only
        assert results.get(key) is not ResultState.VALID
        assert not unresolved(state, key)
        history = accepted_history(state, key)
        if not history:
            continue
        # Resubmission is always explicit, observed, terminal, and acknowledged when unknown.
        assert isinstance(retry, Retry) or key in retry
        for identity, slot in history:
            evidence = observations.get(identity)
            assert evidence is not None and not evidence.allocation.retained
            assert not any(value.active for value in task_states(evidence, slot))
            if S.UNKNOWN in task_states(evidence, slot):
                assert key in ack and not isinstance(retry, Retry)
        if retry is Retry.FAILED:
            identity, slot = history[-1]
            assert task_states(observations[identity], slot)[1] in (S.FAILED, S.CANCELLED)
        if retry is Retry.INCOMPLETE:
            assert results.get(key) is ResultState.MISSING
    # Scheduler evidence outside the observation scope never matters.
    scope = set(observation_scope(state, retry, only))
    narrowed = {identity: value for identity, value in observations.items() if identity in scope}
    again = choose(state, narrowed, results, retry=retry, duplicate_risk=ack, only=only)
    assert again == selection
    if not isinstance(retry, Retry) and not retry:
        assert not scope
    # The selection is recordable as one Attempt (the invariant agrees on retry choices).
    if selection.selected:
        record_intent(
            state,
            allocation_id="f" * 24,
            task_keys=selection.selected,
            profile=profile(),
            retry=selection.retry,
            duplicate_risk=selection.duplicate_risk,
            at=NOW,
        )


@settings(max_examples=1000, deadline=None)
@given(scenarios(), st.sampled_from(list(Retry)), onlys)
def test_bulk_selectors_never_raise(
    scenario: tuple[State, dict[str, Observation], dict[str, ResultState]],
    retry: Retry,
    only: tuple[str, ...] | None,
) -> None:
    state, observations, results = scenario
    choose(state, observations, results, retry=retry, only=only)


@settings(max_examples=1000, deadline=None)
@given(scenarios(), subsets)
def test_acknowledging_retried_keys_never_shrinks_selection(
    scenario: tuple[State, dict[str, Observation], dict[str, ResultState]],
    retry: tuple[str, ...],
) -> None:
    state, observations, results = scenario
    try:
        before = choose(state, observations, results, retry=retry)
    except PlanRefused:
        return
    after = choose(state, observations, results, retry=retry, duplicate_risk=retry)
    assert set(before.selected) <= set(after.selected)
