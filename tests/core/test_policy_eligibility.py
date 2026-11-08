from __future__ import annotations

from collections.abc import Collection, Mapping

import pytest
from core_helpers import NOW, evidence, intend, roster, scenarios, seen, submit
from hypothesis import given, settings
from hypothesis import strategies as st
from support.builders import profile

from servatus.campaign._evidence import Execution, Observation, Observed
from servatus.campaign._policy import Extension, Hold, Request, Retry, Selection, decide, scope
from servatus.campaign._state import AcceptanceState, Attempt, Link, State, record_intent
from servatus.campaign._status import ResultState
from servatus.errors import PlanRefused

R = ResultState
KEY = "task-0"
QUEUED = seen("PENDING")
RUNNING = seen("RUNNING")
SETTLING = seen("COMPLETED", 0)
SUCCEEDED = seen(None, 0)
FAILED = seen(None, 1)
STOPPED = seen(None, stopped=True)
NEVER_STARTED = seen(None, started=False)
LOST = seen(None)


def choose(
    state: State,
    observations: Mapping[str, Observation] | None = None,
    results: dict[str, ResultState] | None = None,
    *,
    strays: Mapping[str, int] | None = None,
    retry: Collection[str] | Retry = (),
    only: Collection[str] | None = None,
) -> Selection:
    request = Request.parse(retry, only)
    return decide(state, evidence(observations, strays), results or {}, request)


def observed_ids(
    state: State, retry: Collection[str] | Retry = (), only: Collection[str] | None = None
) -> tuple[str, ...]:
    return scope(state, Request.parse(retry, only))[1]


def once(observation: Observation) -> tuple[State, dict[str, Observation]]:
    state, identity = submit(roster(1), [KEY])
    return state, {identity: observation}


def outcome(selection: Selection) -> str | Hold:
    """``selected`` or the hold reason for ``KEY``."""
    return "selected" if KEY in selection.selected else selection.held[KEY]


# --- hold order ------------------------------------------------------------------------------


def test_holds_apply_in_their_documented_order() -> None:
    state, first = submit(roster(1), [KEY])
    state, _ = intend(state, [KEY])
    live = {first: RUNNING}
    valid = {KEY: R.VALID}
    assert outcome(choose(state, live, valid, retry=Retry.FAILED, only=())) == Hold.NOT_REQUESTED
    assert outcome(choose(state, live, valid, retry=Retry.FAILED)) == Hold.VALID
    assert outcome(choose(state, live, retry=Retry.FAILED)) == Hold.UNRESOLVED
    accepted, identity = submit(roster(1), [KEY])
    assert outcome(choose(accepted, {identity: RUNNING})) == Hold.SUBMITTED
    assert outcome(choose(accepted, {identity: RUNNING}, retry=Retry.FAILED)) == Hold.ACTIVE
    assert outcome(choose(accepted, {identity: FAILED}, retry=Retry.FAILED)) == "selected"


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
    state, other = submit(roster(2), ["task-1"])
    results = {} if result is None else {KEY: result}
    assert outcome(choose(state, {other: SUCCEEDED}, results, retry=retry)) == expected


def test_not_submitted_history_is_still_fresh_unless_slurm_lists_it() -> None:
    state, identity = submit(roster(1), [KEY], accepted=False)
    assert outcome(choose(state)) == "selected"
    for retry in ((), Retry.FAILED, Retry.INCOMPLETE):
        assert outcome(choose(state, strays={identity: 77}, retry=retry)) == Hold.ACTIVE
    with pytest.raises(PlanRefused, match="retry needs an earlier accepted attempt") as refused:
        choose(state, retry=[KEY])
    assert refused.value.keys == (KEY,)


def test_only_limits_the_plan_and_overrides_retry() -> None:
    state, identity = submit(roster(3), ["task-1"])
    observations = {identity: FAILED}
    selection = choose(state, observations, only=["task-0", "task-1"], retry=["task-1", "task-2"])
    assert selection.selected == ("task-0", "task-1")
    assert dict(selection.held) == {"task-2": Hold.NOT_REQUESTED}
    assert choose(state, only=()).held == {
        "task-0": Hold.NOT_REQUESTED,
        "task-1": Hold.NOT_REQUESTED,
        "task-2": Hold.NOT_REQUESTED,
    }


# --- accepted work without a retry request ---------------------------------------------------


@pytest.mark.parametrize("retry", [(), ["task-1"]])
def test_unrequested_accepted_work_is_submitted_without_observing_it(
    retry: Collection[str],
) -> None:
    state, _ = submit(roster(2), [KEY])
    state, other = submit(state, ["task-1"])
    observations = {other: FAILED} if retry else {}
    assert outcome(choose(state, observations, retry=retry)) == Hold.SUBMITTED
    assert observed_ids(state) == ()


# --- live work -------------------------------------------------------------------------------


@pytest.mark.parametrize("live", [QUEUED, RUNNING, SETTLING], ids=["queued", "running", "settling"])
def test_live_accepted_work_is_active_for_bulk_retry_and_refused_by_key(
    live: Observation,
) -> None:
    state, observations = once(live)
    for retry in (Retry.FAILED, Retry.INCOMPLETE):
        assert outcome(choose(state, observations, {KEY: R.MISSING}, retry=retry)) == Hold.ACTIVE
    with pytest.raises(PlanRefused, match="Slurm still lists earlier work") as refused:
        choose(state, observations, retry=[KEY])
    assert refused.value.keys == (KEY,)


def test_any_live_accepted_attempt_holds_the_task_not_only_the_current_one() -> None:
    state, first = submit(roster(1), [KEY])
    state, second = submit(state, [KEY])
    observations = {first: SETTLING, second: FAILED}
    assert outcome(choose(state, observations, retry=Retry.FAILED)) == Hold.ACTIVE
    with pytest.raises(PlanRefused, match="Slurm still lists earlier work"):
        choose(state, observations, retry=[KEY])
    assert outcome(choose(state, {first: LOST, second: FAILED}, retry=Retry.FAILED)) == "selected"


def test_a_stray_holds_a_retried_task_active() -> None:
    state, first = submit(roster(1), [KEY])
    state, stray = submit(state, [KEY], accepted=False)
    observations = {first: FAILED}
    assert outcome(choose(state, observations, retry=Retry.FAILED)) == "selected"
    strays = {stray: 501}
    assert outcome(choose(state, observations, strays=strays, retry=Retry.FAILED)) == Hold.ACTIVE
    assert outcome(choose(state, observations, strays=strays)) == Hold.SUBMITTED
    with pytest.raises(PlanRefused, match="Slurm still lists earlier work"):
        choose(state, observations, strays=strays, retry=[KEY])


# --- bulk retry ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("observation", "expected"),
    [
        (FAILED, "selected"),
        (STOPPED, "selected"),
        (NEVER_STARTED, "selected"),
        (LOST, Hold.SUBMITTED),
        (SUCCEEDED, Hold.SUBMITTED),
    ],
)
def test_retry_failed_selects_failed_and_stopped_work(
    observation: Observation, expected: str | Hold
) -> None:
    state, observations = once(observation)
    assert outcome(choose(state, observations, retry=Retry.FAILED)) == expected


@pytest.mark.parametrize(
    ("observation", "result", "expected"),
    [
        (SUCCEEDED, R.MISSING, "selected"),
        (FAILED, R.MISSING, "selected"),
        (STOPPED, R.MISSING, "selected"),
        (LOST, R.MISSING, Hold.SUBMITTED),
        (SUCCEEDED, R.UNOBSERVED, Hold.SUBMITTED),
        (SUCCEEDED, None, Hold.SUBMITTED),
        (SUCCEEDED, R.VALID, Hold.VALID),
    ],
)
def test_retry_incomplete_selects_terminal_work_with_missing_results(
    observation: Observation, result: ResultState | None, expected: str | Hold
) -> None:
    state, observations = once(observation)
    results = {} if result is None else {KEY: result}
    assert outcome(choose(state, observations, results, retry=Retry.INCOMPLETE)) == expected


def test_the_current_execution_is_the_latest_accepted_attempts() -> None:
    state, first = submit(roster(1), [KEY])
    state, second = submit(state, [KEY])
    state, _ = submit(state, [KEY], accepted=False)
    assert outcome(choose(state, {first: FAILED, second: SUCCEEDED}, retry=Retry.FAILED)) == (
        Hold.SUBMITTED
    )
    assert outcome(choose(state, {first: SUCCEEDED, second: STOPPED}, retry=Retry.FAILED)) == (
        "selected"
    )


def test_bulk_selection_mixes_fresh_and_resubmitted_work_in_roster_order() -> None:
    state = roster(4)
    state, first = submit(state, ["task-1", "task-2"])
    state, second = submit(state, ["task-3"])
    observations = {first: seen(None, 1, 0), second: STOPPED}
    selection = choose(state, observations, retry=Retry.FAILED)
    assert selection.selected == ("task-0", "task-1", "task-3")
    assert dict(selection.held) == {"task-2": Hold.SUBMITTED}
    with pytest.raises(TypeError, match="does not support item assignment"):
        selection.held["x"] = Hold.VALID  # pyright: ignore[reportIndexIssue]


# --- explicit retry --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("observation", "result"),
    [
        (LOST, None),
        (SUCCEEDED, R.MISSING),
        (SUCCEEDED, R.UNOBSERVED),
        (FAILED, None),
        (NEVER_STARTED, None),
    ],
)
def test_an_explicit_key_selects_any_purged_work_without_a_valid_result(
    observation: Observation, result: ResultState | None
) -> None:
    state, observations = once(observation)
    results = {} if result is None else {KEY: result}
    selection = choose(state, observations, results, retry=[KEY])
    assert selection.selected == (KEY,)


def test_unresolved_work_is_held_even_when_retried() -> None:
    state, first = submit(roster(1), [KEY])
    state, _ = intend(state, [KEY])
    for retry in ([KEY], Retry.FAILED, Retry.INCOMPLETE, ()):
        assert outcome(choose(state, {first: FAILED}, retry=retry)) == Hold.UNRESOLVED
    pending, _ = intend(roster(1), [KEY])
    assert outcome(choose(pending, retry=[KEY])) == Hold.UNRESOLVED
    assert observed_ids(state, [KEY]) == () == observed_ids(state, Retry.FAILED)


def test_valid_results_are_never_resubmitted() -> None:
    state, observations = once(FAILED)
    valid = {KEY: R.VALID}
    with pytest.raises(PlanRefused, match="valid results cannot be retried: 'task-0'"):
        choose(state, observations, valid, retry=[KEY])
    for retry in (Retry.FAILED, Retry.INCOMPLETE, ()):
        assert outcome(choose(state, observations, valid, retry=retry)) == Hold.VALID


def test_all_offending_keys_are_refused_together() -> None:
    state = roster(5)
    state, first = submit(state, ["task-0", "task-1"])
    state, _ = submit(state, ["task-3"], accepted=False)
    state, fourth = submit(state, ["task-4"])
    observations = {first: seen("RUNNING", None, None), fourth: LOST}
    with pytest.raises(PlanRefused, match="^plan refused; ") as refused:
        choose(
            state,
            observations,
            {"task-0": R.VALID},
            retry=["task-0", "task-1", "task-2", "task-3", "task-4", "ghost"],
            only=["task-0", "task-1", "task-2", "task-3", "task-4", "nobody"],
        )
    error = refused.value
    assert error.keys == ("ghost", "nobody", "task-0", "task-2", "task-3", "task-1")
    message = str(error)
    for reason in (
        "unknown Task keys: 'ghost', 'nobody'",
        "valid results cannot be retried: 'task-0'",
        "retry needs an earlier accepted attempt: 'task-2', 'task-3'",
        "Slurm still lists earlier work; wait for it or cancel it: 'task-1'",
    ):
        assert reason in message
    assert "task-4" not in message


# --- observation scope and inputs ------------------------------------------------------------


def test_scope_observes_only_attempts_decide_may_consult() -> None:
    state = roster(5)
    state, first = submit(state, ["task-0", "task-1"])
    state, second = submit(state, ["task-2"], accepted=False)
    state, third = submit(state, ["task-2"])
    state, fourth = submit(state, ["task-0"])
    state, fifth = submit(state, ["task-3"], accepted=False)
    state, _ = intend(state, ["task-4"])
    assert observed_ids(state) == (fifth,)
    assert observed_ids(state, ["task-4"]) == (fifth,)
    assert observed_ids(state, ["task-1"]) == (first, fifth)
    assert observed_ids(state, ["task-0"]) == (first, fourth, fifth)
    assert observed_ids(state, ["task-2", "ghost"]) == (second, third, fifth)
    assert observed_ids(state, Retry.FAILED) == (first, second, third, fourth, fifth)
    assert observed_ids(state, Retry.INCOMPLETE, only=["task-2"]) == (second, third)
    assert observed_ids(state, ["task-0"], only=["task-1"]) == ()


def test_a_plain_plan_without_such_history_observes_nothing() -> None:
    assert observed_ids(roster(2)) == ()
    state, _ = submit(roster(2), [KEY])
    state, _ = intend(state, ["task-1"])
    assert observed_ids(state) == ()
    assert choose(state).selected == ()


# --- properties ------------------------------------------------------------------------------

KEYS = ("task-0", "task-1", "task-2")


subsets = st.sets(st.sampled_from(KEYS)).map(lambda chosen: tuple(k for k in KEYS if k in chosen))
requests = st.one_of(subsets, st.sampled_from(list(Retry)))
onlys = st.none() | subsets


def history(state: State, key: str, acceptance: AcceptanceState) -> list[Attempt]:
    return [
        attempt
        for attempt in state.attempts
        if attempt.acceptance is acceptance and key in attempt.task_keys
    ]


def busy(state: State, observed: Observed, key: str) -> bool:
    """Slurm lists some accepted Attempt of ``key``, or a stray of one recorded as not
    submitted."""
    return any(
        observed.attempts[attempt.allocation_id].live
        for attempt in history(state, key, AcceptanceState.ACCEPTED)
    ) or any(
        attempt.allocation_id in observed.strays
        for attempt in history(state, key, AcceptanceState.NOT_SUBMITTED)
    )


def current(state: State, observed: Observed, key: str) -> Execution:
    latest = history(state, key, AcceptanceState.ACCEPTED)[-1]
    execution, _, _ = observed.attempts[latest.allocation_id].execution(latest.task_keys.index(key))
    return execution


def expected_refusals(
    state: State,
    observed: Observed,
    results: dict[str, ResultState],
    retry: tuple[str, ...] | Retry,
    only: tuple[str, ...] | None,
) -> set[str]:
    refused: set[str] = set()
    for key in () if isinstance(retry, Retry) else retry:
        if only is not None and key not in only:
            continue
        if results.get(key) is ResultState.VALID:
            refused.add(key)
        elif history(state, key, AcceptanceState.UNRESOLVED):
            continue
        elif not history(state, key, AcceptanceState.ACCEPTED) or busy(state, observed, key):
            refused.add(key)
    return refused


@settings(max_examples=2000, deadline=None)
@given(scenarios(every_accepted_observed=True), requests, onlys)
def test_policy_safety_properties(
    scenario: tuple[State, Observed, dict[str, ResultState]],
    retry: tuple[str, ...] | Retry,
    only: tuple[str, ...] | None,
) -> None:
    state, observed, results = scenario
    expected = expected_refusals(state, observed, results, retry, only)
    try:
        selection = decide(state, observed, results, Request.parse(retry, only))
    except PlanRefused as refusal:
        assert set(refusal.keys) == expected and expected
        assert len(refusal.keys) == len(set(refusal.keys))
        return
    assert not expected
    assert set(selection.selected).isdisjoint(selection.held)
    assert set(selection.selected) | set(selection.held) == set(KEYS)
    assert selection.selected == tuple(key for key in KEYS if key in selection.selected)
    for key in selection.selected:
        assert only is None or key in only
        assert results.get(key) is not ResultState.VALID
        assert not history(state, key, AcceptanceState.UNRESOLVED)
        assert not busy(state, observed, key)
        if not history(state, key, AcceptanceState.ACCEPTED):
            continue
        assert isinstance(retry, Retry) or key in retry
        if retry is Retry.FAILED:
            assert current(state, observed, key) in (Execution.FAILED, Execution.STOPPED)
        if retry is Retry.INCOMPLETE:
            assert current(state, observed, key).terminal
            assert results.get(key) is ResultState.MISSING
    request = Request.parse(retry, only)
    probed, consulted = scope(state, request)
    narrowed = Observed(
        {name: value for name, value in observed.attempts.items() if name in consulted},
        {name: job for name, job in observed.strays.items() if name in consulted},
    )
    assert decide(state, narrowed, results, request) == selection
    asked = {task.key for task in probed}
    answered = {key: value for key, value in results.items() if key in asked}
    # An unprobed Task can only change from VALID to SUBMITTED; it is never selectable.
    assert decide(state, observed, answered, request).selected == selection.selected
    if selection.selected:
        record_intent(
            state,
            allocation_id="f" * 24,
            link=Link("f" * 24, 1, 1),
            task_keys=selection.selected,
            profile=profile(),
            at=NOW,
        )


@settings(max_examples=1000, deadline=None)
@given(scenarios(every_accepted_observed=True), st.sampled_from(list(Retry)), onlys)
def test_bulk_selectors_never_raise(
    scenario: tuple[State, Observed, dict[str, ResultState]],
    retry: Retry,
    only: tuple[str, ...] | None,
) -> None:
    state, observed, results = scenario
    decide(state, observed, results, Request.parse(retry, only))


@settings(max_examples=1000, deadline=None)
@given(
    scenarios(every_accepted_observed=True),
    st.sampled_from(list(Retry)) | st.lists(st.sampled_from(KEYS), unique=True).map(tuple),
    st.sets(st.sampled_from(KEYS)),
)
def test_a_budget_never_reruns_spent_work_and_only_named_keys_extend(
    scenario: tuple[State, Observed, dict[str, ResultState]],
    retry: tuple[str, ...] | Retry,
    exhausted: set[str],
) -> None:
    state, observed, results = scenario
    request = Request.parse(retry)
    budgeted = Request(request.retry, request.only, frozenset(exhausted))
    try:
        selection = decide(state, observed, results, budgeted)
    except PlanRefused as refusal:
        assert not isinstance(retry, Retry) and refusal.keys
        return
    for key in selection.selected:
        if history(state, key, AcceptanceState.ACCEPTED):
            assert key not in exhausted
    assert set(selection.extensions) <= set(selection.selected)
    assert isinstance(retry, Retry) <= (not selection.extensions)
    for key, reason in selection.extensions.items():
        valid = results.get(key) is ResultState.VALID
        assert reason is (Extension.VALID if valid else Extension.SUCCEEDED)
    assert all(hold is not Hold.SPENT or key in exhausted for key, hold in selection.held.items())
