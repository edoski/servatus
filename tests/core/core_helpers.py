"""Synthetic Campaign histories and scheduler evidence for the core tests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta

from hypothesis import strategies as st
from support.builders import profile, tasks

from servatus.campaign import _codec
from servatus.campaign._config import Profile
from servatus.campaign._evidence import Observation, Observed, StepEnd
from servatus.campaign._state import (
    AcceptanceState,
    Link,
    State,
    create,
    record_intent,
    record_outcome,
)
from servatus.campaign._status import ResultState

NOW = datetime(2030, 1, 1, tzinfo=UTC)
CAMPAIGN_ID = "c" * 32


def allocation_id(number: int) -> str:
    return f"{number:024x}"


def roster(count: int = 3, *, appendable: bool = False) -> State:
    return create(CAMPAIGN_ID, tasks(count), appendable=appendable)


def intend(
    state: State,
    keys: Iterable[str],
    *,
    profile_value: Profile | None = None,
    link: Link | None = None,
) -> tuple[State, str]:
    """Record intent for ``keys``, as link 1 of 1 unless ``link`` says otherwise; the identity
    numbers the Attempt."""
    identity = allocation_id(len(state.attempts) + 1)
    changed = record_intent(
        state,
        allocation_id=identity,
        link=link or Link(identity, 1, 1),
        task_keys=tuple(keys),
        profile=profile_value or profile(),
        at=NOW + timedelta(minutes=len(state.attempts)),
    )
    return changed, identity


def submit(
    state: State,
    keys: Iterable[str],
    *,
    accepted: bool = True,
    profile_value: Profile | None = None,
    link: Link | None = None,
) -> tuple[State, str]:
    """Intent plus outcome: job ``100 + attempt count`` when accepted, else non-submission."""
    changed, identity = intend(state, keys, profile_value=profile_value, link=link)
    job = len(changed.attempts) + 100 if accepted else None
    return record_outcome(changed, identity, job), identity


def seen(
    slurm_state: str | None = None,
    *exit_codes: int | None,
    started: bool = True,
    stopped: bool = False,
    limited: bool = False,
    handed: tuple[bool, ...] | None = None,
) -> Observation:
    """One allocation's evidence: its squeue state (``None`` once purged), markers, and one exit
    code per slot (a single unfinished slot by default). A started allocation started at ``NOW``;
    steps return 10 minutes later, after the pre-limit signal (5 minutes) when ``limited``, and a
    stop comes 20 minutes later."""
    codes = exit_codes or (None,)
    return Observation(
        slurm_state,
        NOW if started else None,
        NOW + timedelta(minutes=20) if stopped else None,
        NOW + timedelta(minutes=5) if limited else None,
        tuple(
            None if code is None else StepEnd(code, NOW + timedelta(minutes=10)) for code in codes
        ),
        handed or (True,) * len(codes),
    )


def evidence(
    attempts: Mapping[str, Observation] | None = None, strays: Mapping[str, int] | None = None
) -> Observed:
    return Observed(dict(attempts or {}), dict(strays or {}))


def reencode(value: object) -> bytes:
    """Canonical bytes of an edited state document, as the store would have written them."""
    return _codec.canonical(value) + b"\n"


def unresolved_keys(state: State) -> set[str]:
    """Keys an unresolved Attempt holds; no new intent may name them."""
    return {
        key
        for attempt in state.attempts
        if attempt.outcome_revision is None
        for key in attempt.task_keys
    }


SLURM_STATES = st.sampled_from([None, None, "PENDING", "RUNNING", "COMPLETED", "NODE_FAIL"])


@st.composite
def observations_of(draw: st.DrawFn, task_count: int) -> Observation:
    return seen(
        draw(SLURM_STATES),
        *(draw(st.none() | st.integers(0, 255)) for _ in range(task_count)),
        started=draw(st.booleans()),
        stopped=draw(st.booleans()),
        limited=draw(st.booleans()),
    )


@st.composite
def scenarios(
    draw: st.DrawFn, *, every_accepted_observed: bool
) -> tuple[State, Observed, dict[str, ResultState]]:
    """A three-Task history with evidence for its accepted Attempts (or only some of them),
    strays for some not-submitted ones, and probe results for some Tasks."""
    state = roster(3)
    for _ in range(draw(st.integers(0, 5))):
        free = [task.key for task in state.tasks if task.key not in unresolved_keys(state)]
        if not free:
            break
        chosen = draw(st.lists(st.sampled_from(free), min_size=1, max_size=3, unique=True))
        keys = [key for key in free if key in chosen]
        kind = draw(st.sampled_from(["accept", "accept", "reject", "pending"]))
        if kind == "pending":
            state, _ = intend(state, keys)
        else:
            state, _ = submit(state, keys, accepted=kind == "accept")
    attempts: dict[str, Observation] = {}
    strays: dict[str, int] = {}
    for attempt in state.attempts:
        if attempt.acceptance is AcceptanceState.ACCEPTED:
            if every_accepted_observed or draw(st.integers(0, 4)):
                attempts[attempt.allocation_id] = draw(observations_of(len(attempt.task_keys)))
        elif attempt.acceptance is AcceptanceState.NOT_SUBMITTED and draw(st.booleans()):
            strays[attempt.allocation_id] = 9000
    results = {
        task.key: draw(st.sampled_from(list(ResultState)))
        for task in state.tasks
        if draw(st.booleans())
    }
    return state, Observed(attempts, strays), results
