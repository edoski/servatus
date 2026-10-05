"""The eligibility policy: which Tasks one plan may submit, as a pure decision.

Every Task in the roster ends up either selected or held with one ``Hold`` reason, evaluated in
this order:

1. ``only`` given and the Task is not in it: ``NOT_REQUESTED``.
2. The result probe said VALID: ``VALID``. Valid work is never resubmitted.
3. An Attempt naming the Task has unresolved acceptance: ``UNRESOLVED``. Reconcile it first.
4. No accepted Attempt has ever named the Task: selected as fresh work.
5. The Task has accepted work and resubmission was not requested for it: ``SUBMITTED``. Its
   scheduler state is not consulted (the work may still be queued, running, or long finished),
   so plans without retries do not depend on (or observe) scheduler evidence.
6. Any accepted Attempt is queued, running, or retained by the scheduler: ``ACTIVE``.
7. Any accepted Attempt lacks scheduler evidence: ``UNOBSERVABLE``.
8. Otherwise every accepted Attempt is terminal or UNKNOWN, and the retry selector decides.

Resubmission is always explicit, in one of two forms:

- Explicit keys (``retry=["a", "b"]``) select each named Task. An UNKNOWN Attempt (from the
  allocation, or from the Task's own step) additionally needs the key in ``duplicate_risk``.
  Explicit keys are checked strictly: unknown keys, VALID results, Tasks with no accepted
  history, UNKNOWN evidence without acknowledgement, and acknowledgements of keys that were not
  explicitly retried are all collected into one ``PlanRefused`` listing every offending key.
  Explicit retries of active, unresolved, or unobservable Tasks are held, not refused.
- A bulk selector never raises for an individual Task. ``Retry.FAILED`` selects Tasks whose
  current (latest accepted) Attempt failed or was cancelled, judged by the Task's step when
  known and by the allocation otherwise. ``Retry.INCOMPLETE`` selects terminal Tasks whose
  result was observed MISSING; it needs a probe, and unprobed Tasks are held ``UNOBSERVABLE``.
  UNKNOWN evidence holds a Task as ``UNOBSERVABLE``: acknowledging duplicate risk always
  requires explicit keys.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum
from types import MappingProxyType
from typing import cast

from ..errors import ConfigurationError, PlanRefused
from ._evidence import AllocationState, Observation
from ._state import AcceptanceState, Attempt, State
from ._status import ResultState, task_execution


class Hold(StrEnum):
    """Why a plan does not submit a Task.

    ``VALID``: the probe reports a valid result. ``UNRESOLVED``: an intent has no recorded
    outcome. ``ACTIVE``: the scheduler still holds earlier work. ``SUBMITTED``: the Task has
    accepted work and was not selected for retry (its scheduler state was not consulted).
    ``UNOBSERVABLE``: evidence for earlier work is missing or unknown. ``NOT_REQUESTED``:
    excluded by ``only``.
    """

    VALID = "VALID"
    UNRESOLVED = "UNRESOLVED"
    ACTIVE = "ACTIVE"
    SUBMITTED = "SUBMITTED"
    UNOBSERVABLE = "UNOBSERVABLE"
    NOT_REQUESTED = "NOT_REQUESTED"


class Retry(Enum):
    """Bulk resubmission selectors."""

    FAILED = "FAILED"
    INCOMPLETE = "INCOMPLETE"


@dataclass(frozen=True, slots=True)
class Selection:
    """Selected Task keys in roster order, the reason for every other Task, and the subsets of
    ``selected`` that resubmit accepted work (``retry``) under acknowledged ``duplicate_risk``."""

    selected: tuple[str, ...]
    held: Mapping[str, Hold]
    retry: tuple[str, ...]
    duplicate_risk: tuple[str, ...]


_REFUSALS = (
    ("unknown", "unknown Task keys"),
    ("valid", "valid results cannot be retried"),
    ("unaccepted", "retry needs an earlier accepted attempt"),
    ("unknown_evidence", "UNKNOWN scheduler evidence needs a duplicate-risk acknowledgement"),
    ("unretried", "a duplicate-risk acknowledgement requires an explicitly retried Task key"),
)


def _keys(value: object, name: str) -> frozenset[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Collection):
        raise ConfigurationError(f"{name} must be a collection of Task keys")
    items = tuple(cast(Collection[object], value))
    if any(not isinstance(item, str) for item in items):
        raise ConfigurationError(f"{name} must contain only Task key strings")
    return frozenset(cast(tuple[str, ...], items))


def _explicit(retry: Collection[str] | Retry) -> frozenset[str] | None:
    return None if isinstance(retry, Retry) else _keys(retry, "retry")


def _history(state: State) -> tuple[set[str], dict[str, list[tuple[Attempt, int]]]]:
    unresolved: set[str] = set()
    accepted: dict[str, list[tuple[Attempt, int]]] = {}
    for attempt in state.attempts:
        if attempt.acceptance is AcceptanceState.UNRESOLVED:
            unresolved.update(attempt.task_keys)
        elif attempt.acceptance is AcceptanceState.ACCEPTED:
            for slot, key in enumerate(attempt.task_keys):
                accepted.setdefault(key, []).append((attempt, slot))
    return unresolved, accepted


def observation_scope(
    state: State, retry: Collection[str] | Retry = (), only: Collection[str] | None = None
) -> tuple[str, ...]:
    """Allocation ids whose evidence ``decide`` may consult for these requests, in state order.

    Only accepted Attempts of Tasks that could be retried are included, so a plan that selects
    only fresh work observes nothing. Unknown keys are ignored here; ``decide`` refuses them.
    """
    explicit = _explicit(retry)
    scope = None if only is None else _keys(only, "only")
    unresolved, accepted = _history(state)
    wanted: set[str] = set()
    for key, history in accepted.items():
        if (
            (scope is None or key in scope)
            and (explicit is None or key in explicit)
            and key not in unresolved
        ):
            wanted.update(attempt.allocation_id for attempt, _ in history)
    return tuple(
        attempt.allocation_id for attempt in state.attempts if attempt.allocation_id in wanted
    )


def decide(
    state: State,
    observations: Mapping[str, Observation],
    results: Mapping[str, ResultState],
    *,
    retry: Collection[str] | Retry = (),
    duplicate_risk: Collection[str] = (),
    only: Collection[str] | None = None,
) -> Selection:
    """Apply the eligibility policy in the module docstring to every Task in the roster.

    ``observations`` maps allocation ids to scheduler evidence (see ``observation_scope``);
    ``results`` maps Task keys to probe results, with absent keys UNOBSERVED.
    """
    explicit = _explicit(retry)
    acknowledged = _keys(duplicate_risk, "duplicate_risk")
    scope = None if only is None else _keys(only, "only")
    roster = tuple(task.key for task in state.tasks)
    known = set(roster)
    refused: dict[str, list[str]] = {name: [] for name, _ in _REFUSALS}
    mentioned = (explicit or frozenset()) | acknowledged | (scope or frozenset())
    refused["unknown"] = sorted(mentioned - known)
    refused["unretried"] = [
        key for key in roster if key in acknowledged and (explicit is None or key not in explicit)
    ]
    unresolved, accepted = _history(state)
    selected: list[str] = []
    held: dict[str, Hold] = {}
    retried: list[str] = []
    for key in roster:
        named = explicit is not None and key in explicit
        if scope is not None and key not in scope:
            held[key] = Hold.NOT_REQUESTED
            continue
        result = results.get(key, ResultState.UNOBSERVED)
        if result is ResultState.VALID:
            if named:
                refused["valid"].append(key)
            held[key] = Hold.VALID
            continue
        if key in unresolved:
            held[key] = Hold.UNRESOLVED
            continue
        history = accepted.get(key, [])
        if not history:
            if named:
                refused["unaccepted"].append(key)
            else:
                selected.append(key)
            continue
        if explicit is not None and not named:
            held[key] = Hold.SUBMITTED
            continue
        hold, unknown, current = _assess(history, observations)
        if hold is not None:
            held[key] = hold
        elif named:
            if unknown and key not in acknowledged:
                refused["unknown_evidence"].append(key)
            else:
                selected.append(key)
                retried.append(key)
        elif unknown:
            held[key] = Hold.UNOBSERVABLE
        elif retry is Retry.FAILED:
            if current in (AllocationState.FAILED, AllocationState.CANCELLED):
                selected.append(key)
                retried.append(key)
            else:
                held[key] = Hold.SUBMITTED
        elif result is ResultState.MISSING:
            selected.append(key)
            retried.append(key)
        else:
            held[key] = Hold.UNOBSERVABLE
    if any(refused.values()):
        _refuse(refused)
    return Selection(
        selected=tuple(selected),
        held=MappingProxyType(held),
        retry=tuple(retried),
        duplicate_risk=tuple(key for key in retried if key in acknowledged),
    )


def _assess(
    history: list[tuple[Attempt, int]], observations: Mapping[str, Observation]
) -> tuple[Hold | None, bool, AllocationState | None]:
    """Hold reason (if any), whether any Attempt is UNKNOWN, and the current execution state."""
    unobserved = active = unknown = False
    current: AllocationState | None = None
    for attempt, slot in history:
        observation = observations.get(attempt.allocation_id)
        if observation is None:
            unobserved = True
            continue
        allocation = observation.allocation
        current, _ = task_execution(observation, slot)
        active = active or allocation.retained or allocation.state.active or current.active
        unknown = unknown or AllocationState.UNKNOWN in (allocation.state, current)
    if active:
        return Hold.ACTIVE, unknown, current
    if unobserved:
        return Hold.UNOBSERVABLE, unknown, current
    return None, unknown, current


def _refuse(refused: Mapping[str, list[str]]) -> None:
    reasons = [
        f"{label}: {', '.join(map(repr, refused[name]))}"
        for name, label in _REFUSALS
        if refused[name]
    ]
    keys = tuple(dict.fromkeys(key for name, _ in _REFUSALS for key in refused[name]))
    raise PlanRefused("plan refused; " + "; ".join(reasons), keys=keys)
