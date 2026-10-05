"""Synthetic Campaign histories and scheduler evidence for the core tests."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

from support.builders import profile, tasks

from servatus.campaign import _codec
from servatus.campaign._config import Profile
from servatus.campaign._evidence import (
    AllocationState,
    JobRef,
    Observation,
    SchedulerEvidence,
    StepEvidence,
)
from servatus.campaign._state import State, create, record_intent, record_outcome

NOW = datetime(2030, 1, 1, tzinfo=UTC)
CAMPAIGN_ID = "c" * 32


def allocation_id(number: int) -> str:
    return f"{number:024x}"


def roster(count: int = 3, *, appendable: bool = False) -> State:
    return create(CAMPAIGN_ID, tasks(count), appendable=appendable)


def accepted_keys(state: State) -> set[str]:
    return {
        key
        for attempt in state.attempts
        if attempt.acceptance.value == "ACCEPTED"
        for key in attempt.task_keys
    }


def intend(
    state: State,
    keys: Iterable[str],
    *,
    ack: Iterable[str] = (),
    profile_value: Profile | None = None,
) -> tuple[State, str]:
    """Record intent for ``keys`` with the retry choices the invariant demands."""
    chosen, acknowledged = tuple(keys), set(ack)
    prior = accepted_keys(state)
    retry = tuple(key for key in chosen if key in prior)
    identity = allocation_id(len(state.attempts) + 1)
    changed, _ = record_intent(
        state,
        allocation_id=identity,
        task_keys=chosen,
        profile=profile_value or profile(),
        retry=retry,
        duplicate_risk=tuple(key for key in retry if key in acknowledged),
        at=NOW + timedelta(minutes=len(state.attempts)),
    )
    return changed, identity


def submit(
    state: State,
    keys: Iterable[str],
    *,
    job: int | None = 0,
    ack: Iterable[str] = (),
    profile_value: Profile | None = None,
) -> tuple[State, str]:
    """Intent plus outcome: ``job=0`` picks a fresh job number, ``None`` records non-submission."""
    changed, identity = intend(state, keys, ack=ack, profile_value=profile_value)
    number = len(changed.attempts) + 100 if job == 0 else job
    reference = None if number is None else JobRef(number)
    return record_outcome(changed, identity, reference), identity


def observed(
    state: AllocationState = AllocationState.SUCCEEDED,
    *,
    retained: bool = False,
    exit_code: str | None = None,
    steps: Iterable[StepEvidence | None] = (),
) -> Observation:
    return Observation(
        SchedulerEvidence(state, raw_state=state.value, exit_code=exit_code, retained=retained),
        tuple(steps),
    )


def step(state: AllocationState, exit_code: str | None = None) -> StepEvidence:
    return StepEvidence(state, state.value, exit_code)


def document(data: bytes) -> dict[str, object]:
    return json.loads(data)


def reencode(value: object) -> bytes:
    """Canonical bytes of an edited state document, as the store would have written them."""
    return _codec.canonical(value) + b"\n"
