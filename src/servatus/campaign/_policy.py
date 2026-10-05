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
  current (latest accepted) Attempt failed or was cancelled, judged by ``task_execution``: the
  Task's step when known, else the allocation (a failed packed allocation without the Task's
  step is UNKNOWN, because that Task may have succeeded). ``Retry.INCOMPLETE`` selects terminal
  Tasks whose result was observed MISSING; it needs a probe, and unprobed Tasks are held
  ``UNOBSERVABLE``.
  UNKNOWN evidence holds a Task as ``UNOBSERVABLE``: acknowledging duplicate risk always
  requires explicit keys.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum
from types import MappingProxyType

from ..errors import PlanRefused
from ._config import Task, task_keys
from ._evidence import AllocationState, Observation
from ._state import AcceptanceState, Attempt, State
from ._status import ResultState, own_step, task_execution


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


def _explicit(retry: Collection[str] | Retry) -> frozenset[str] | None:
    return None if isinstance(retry, Retry) else frozenset(task_keys(retry, "retry"))


def _scope(only: Collection[str] | None) -> frozenset[str] | None:
    return None if only is None else frozenset(task_keys(only, "only"))


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


def result_scope(
    state: State, retry: Collection[str] | Retry = (), only: Collection[str] | None = None
) -> tuple[Task, ...]:
    """Tasks whose probe result can change ``decide``: never accepted, or eligible for retry."""
    explicit, scope, accepted = _explicit(retry), _scope(only), state.accepted_keys
    return tuple(
        task
        for task in state.tasks
        if (scope is None or task.key in scope)
        and (task.key not in accepted or explicit is None or task.key in explicit)
    )


def observation_scope(
    state: State, retry: Collection[str] | Retry = (), only: Collection[str] | None = None
) -> tuple[str, ...]:
    """Allocation ids whose evidence ``decide`` may consult for these requests, in state order.

    Only accepted Attempts of Tasks that could be retried are included, so a plan that selects
    only fresh work observes nothing. Unknown keys are ignored here; ``decide`` refuses them.
    """
    explicit, scope = _explicit(retry), _scope(only)
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
    explicit, scope = _explicit(retry), _scope(only)
    acknowledged = frozenset(task_keys(duplicate_risk, "duplicate_risk"))
    roster = tuple(task.key for task in state.tasks)
    refused: dict[str, list[str]] = {name: [] for name, _ in _REFUSALS}
    mentioned = (explicit or frozenset()) | acknowledged | (scope or frozenset())
    refused["unknown"] = sorted(mentioned - set(roster))
    refused["unretried"] = [
        key for key in roster if key in acknowledged and key not in (explicit or ())
    ]
    unresolved, accepted = _history(state)
    selected: list[str] = []
    held: dict[str, Hold] = {}
    for key in roster:
        if scope is not None and key not in scope:
            outcome: Hold | str | None = Hold.NOT_REQUESTED
        else:
            outcome = _judge(
                accepted.get(key, []),
                observations,
                results.get(key, ResultState.UNOBSERVED),
                named=explicit is not None and key in explicit,
                bulk=retry if isinstance(retry, Retry) else None,
                unresolved=key in unresolved,
                acknowledged=key in acknowledged,
            )
        if outcome is None:
            selected.append(key)
        elif isinstance(outcome, Hold):
            held[key] = outcome
        else:
            refused[outcome].append(key)
    if any(refused.values()):
        _refuse(refused)
    retried = tuple(key for key in selected if key in accepted)
    return Selection(
        selected=tuple(selected),
        held=MappingProxyType(held),
        retry=retried,
        duplicate_risk=tuple(key for key in retried if key in acknowledged),
    )


def _judge(
    history: list[tuple[Attempt, int]],
    observations: Mapping[str, Observation],
    result: ResultState,
    *,
    named: bool,
    bulk: Retry | None,
    unresolved: bool,
    acknowledged: bool,
) -> Hold | str | None:
    """One requested Task: its hold reason, a refusal name (explicit keys only), or None to
    select it. ``history`` lists its accepted Attempts with its slot in each."""
    if result is ResultState.VALID:
        return "valid" if named else Hold.VALID
    if unresolved:
        return Hold.UNRESOLVED
    if not history:
        return "unaccepted" if named else None
    if not named and bulk is None:
        return Hold.SUBMITTED
    hold, unknown, current = _assess(history, observations)
    if hold is not None:
        return hold
    if named:
        return "unknown_evidence" if unknown and not acknowledged else None
    if unknown:
        return Hold.UNOBSERVABLE
    if bulk is Retry.FAILED:
        failed = current in (AllocationState.FAILED, AllocationState.CANCELLED)
        return None if failed else Hold.SUBMITTED
    return None if result is ResultState.MISSING else Hold.UNOBSERVABLE


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
        allocation, step = observation.allocation, own_step(observation, slot)
        current, _ = task_execution(observation, slot, len(attempt.task_keys))
        active = (
            active
            or allocation.retained
            or allocation.state.active
            or current.active
            or (step is not None and step.state.active)
        )
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
