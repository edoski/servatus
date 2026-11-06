"""Synthetic Campaign histories and scheduler evidence for the core tests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta

from support.builders import profile, tasks

from servatus.campaign import _codec
from servatus.campaign._config import Profile
from servatus.campaign._evidence import Observation, Observed
from servatus.campaign._state import State, create, record_intent, record_outcome

NOW = datetime(2030, 1, 1, tzinfo=UTC)
CAMPAIGN_ID = "c" * 32


def allocation_id(number: int) -> str:
    return f"{number:024x}"


def roster(count: int = 3, *, appendable: bool = False) -> State:
    return create(CAMPAIGN_ID, tasks(count), appendable=appendable)


def intend(
    state: State, keys: Iterable[str], *, profile_value: Profile | None = None
) -> tuple[State, str]:
    """Record intent for ``keys``; the identity numbers the Attempt."""
    identity = allocation_id(len(state.attempts) + 1)
    changed = record_intent(
        state,
        allocation_id=identity,
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
) -> tuple[State, str]:
    """Intent plus outcome: job ``100 + attempt count`` when accepted, else non-submission."""
    changed, identity = intend(state, keys, profile_value=profile_value)
    job = len(changed.attempts) + 100 if accepted else None
    return record_outcome(changed, identity, job), identity


def seen(
    slurm_state: str | None = None,
    *exit_codes: int | None,
    started: bool = True,
    stopped: bool = False,
) -> Observation:
    """One allocation's evidence: its squeue state (``None`` once purged), markers, and one exit
    code per slot (a single unfinished slot by default)."""
    return Observation(slurm_state, started, stopped, exit_codes or (None,))


def evidence(
    attempts: Mapping[str, Observation] | None = None, strays: Mapping[str, int] | None = None
) -> Observed:
    return Observed(dict(attempts or {}), dict(strays or {}))


def reencode(value: object) -> bytes:
    """Canonical bytes of an edited state document, as the store would have written them."""
    return _codec.canonical(value) + b"\n"
