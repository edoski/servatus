"""The eligibility policy: which Tasks one plan may submit, as a pure decision.

Every Task in the roster ends up either selected or held with one ``Hold`` reason, evaluated in
this order:

1. ``only`` given and the Task is not in it: ``NOT_REQUESTED``.
2. The result probe said VALID: ``VALID``. Valid work is never resubmitted.
3. An Attempt naming the Task has unresolved acceptance: ``UNRESOLVED``. Reconcile it first.
4. No accepted Attempt has ever named the Task: selected as fresh work, unless Slurm still lists
   a job of an allocation recorded as not submitted for it (``ACTIVE``).
5. The Task has accepted work and resubmission was not requested for it: ``SUBMITTED``. Its
   scheduler state is not consulted (the work may still be queued, running, or long finished),
   so plans without retries do not depend on (or observe) scheduler evidence.
6. Slurm still lists a job of any accepted (or not-submitted) allocation of the Task, settling
   ones included: ``ACTIVE``.
7. Otherwise the retry selector decides from the Task's current (latest accepted) execution.

Resubmission is always explicit, in one of two forms:

- Explicit keys (``retry=["a", "b"]``) select each named Task, whatever its current execution,
  ``LOST`` included: naming the key is the operator's acknowledgement. They are checked strictly:
  unknown keys, VALID results, Tasks with no accepted history, and Tasks whose earlier work Slurm
  still lists are collected into one ``PlanRefused`` listing every offending key.
- A bulk selector never raises for an individual Task. ``Retry.FAILED`` selects Tasks whose
  current execution is ``FAILED`` or ``STOPPED``, never ``LOST``. ``Retry.INCOMPLETE`` selects
  Tasks whose current execution is terminal and whose result was observed MISSING; it needs a
  probe. Every other Task stays ``SUBMITTED``.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum
from types import MappingProxyType

from ..errors import PlanRefused
from ._config import Task, task_keys
from ._evidence import Execution, Observed
from ._state import AcceptanceState, Attempt, State
from ._status import ResultState


class Hold(StrEnum):
    """Why a plan does not submit a Task.

    ``VALID``: the probe reports a valid result. ``UNRESOLVED``: an intent has no recorded
    outcome. ``ACTIVE``: Slurm still lists earlier work. ``SUBMITTED``: the Task has accepted
    work and was not selected for retry. ``NOT_REQUESTED``: excluded by ``only``.
    """

    VALID = "VALID"
    UNRESOLVED = "UNRESOLVED"
    ACTIVE = "ACTIVE"
    SUBMITTED = "SUBMITTED"
    NOT_REQUESTED = "NOT_REQUESTED"


class Retry(Enum):
    """Bulk resubmission selectors."""

    FAILED = "FAILED"
    INCOMPLETE = "INCOMPLETE"


@dataclass(frozen=True, slots=True)
class Selection:
    """Selected Task keys in roster order, the reason for every other Task, and the subset of
    ``selected`` that resubmits accepted work (``retry``)."""

    selected: tuple[str, ...]
    held: Mapping[str, Hold]
    retry: tuple[str, ...]


_REFUSALS = (
    ("unknown", "unknown Task keys"),
    ("valid", "valid results cannot be retried"),
    ("unaccepted", "retry needs an earlier accepted attempt"),
    ("live", "Slurm still lists earlier work; wait for it or cancel it"),
)


def _explicit(retry: Collection[str] | Retry) -> frozenset[str] | None:
    return None if isinstance(retry, Retry) else frozenset(task_keys(retry, "retry"))


def _scope(only: Collection[str] | None) -> frozenset[str] | None:
    return None if only is None else frozenset(task_keys(only, "only"))


@dataclass(frozen=True, slots=True)
class _History:
    """Each Task's unresolved, accepted (with its slot), and not-submitted Attempts."""

    unresolved: frozenset[str]
    accepted: Mapping[str, list[tuple[Attempt, int]]]
    not_submitted: Mapping[str, list[Attempt]]


def _history(state: State) -> _History:
    unresolved: set[str] = set()
    accepted: dict[str, list[tuple[Attempt, int]]] = {}
    not_submitted: dict[str, list[Attempt]] = {}
    for attempt in state.attempts:
        if attempt.acceptance is AcceptanceState.UNRESOLVED:
            unresolved.update(attempt.task_keys)
        elif attempt.acceptance is AcceptanceState.ACCEPTED:
            for slot, key in enumerate(attempt.task_keys):
                accepted.setdefault(key, []).append((attempt, slot))
        else:
            for key in attempt.task_keys:
                not_submitted.setdefault(key, []).append(attempt)
    return _History(frozenset(unresolved), accepted, not_submitted)


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

    Only the Attempts of Tasks that could be selected are included: accepted and not-submitted
    Attempts of retry candidates, and not-submitted Attempts of Tasks never accepted. A plan of
    fresh work without such history observes nothing. Unknown keys are ignored here; ``decide``
    refuses them.
    """
    explicit, scope = _explicit(retry), _scope(only)
    history = _history(state)
    wanted: set[str] = set()
    for task in state.tasks:
        key = task.key
        if (scope is not None and key not in scope) or key in history.unresolved:
            continue
        accepted = history.accepted.get(key, [])
        if accepted and explicit is not None and key not in explicit:
            continue
        wanted.update(attempt.allocation_id for attempt, _ in accepted)
        wanted.update(attempt.allocation_id for attempt in history.not_submitted.get(key, []))
    return tuple(
        attempt.allocation_id for attempt in state.attempts if attempt.allocation_id in wanted
    )


def decide(
    state: State,
    observed: Observed,
    results: Mapping[str, ResultState],
    *,
    retry: Collection[str] | Retry = (),
    only: Collection[str] | None = None,
) -> Selection:
    """Apply the eligibility policy in the module docstring to every Task in the roster.

    ``observed`` must cover ``observation_scope``; ``results`` maps Task keys to probe results,
    with absent keys UNOBSERVED.
    """
    explicit, scope = _explicit(retry), _scope(only)
    roster = tuple(task.key for task in state.tasks)
    refused: dict[str, list[str]] = {name: [] for name, _ in _REFUSALS}
    mentioned = (explicit or frozenset()) | (scope or frozenset())
    refused["unknown"] = sorted(mentioned - set(roster))
    history = _history(state)
    selected: list[str] = []
    held: dict[str, Hold] = {}
    for key in roster:
        if scope is not None and key not in scope:
            outcome: Hold | str | None = Hold.NOT_REQUESTED
        else:
            outcome = _judge(
                key,
                history,
                observed,
                results.get(key, ResultState.UNOBSERVED),
                named=explicit is not None and key in explicit,
                bulk=retry if isinstance(retry, Retry) else None,
            )
        if outcome is None:
            selected.append(key)
        elif isinstance(outcome, Hold):
            held[key] = outcome
        else:
            refused[outcome].append(key)
    if any(refused.values()):
        _refuse(refused)
    return Selection(
        selected=tuple(selected),
        held=MappingProxyType(held),
        retry=tuple(key for key in selected if key in history.accepted),
    )


def _judge(
    key: str,
    history: _History,
    observed: Observed,
    result: ResultState,
    *,
    named: bool,
    bulk: Retry | None,
) -> Hold | str | None:
    """One requested Task: its hold reason, a refusal name (explicit keys only), or None to
    select it."""
    if result is ResultState.VALID:
        return "valid" if named else Hold.VALID
    if key in history.unresolved:
        return Hold.UNRESOLVED
    accepted = history.accepted.get(key, [])
    strays = history.not_submitted.get(key, [])
    stray = any(item.allocation_id in observed.strays for item in strays)
    if not accepted:
        if named:
            return "unaccepted"
        return Hold.ACTIVE if stray else None
    if not named and bulk is None:
        return Hold.SUBMITTED
    if stray or any(observed.attempts[attempt.allocation_id].live for attempt, _ in accepted):
        return "live" if named else Hold.ACTIVE
    if named:
        return None
    attempt, slot = accepted[-1]
    current, _ = observed.attempts[attempt.allocation_id].execution(slot)
    if bulk is Retry.FAILED:
        chosen = current in (Execution.FAILED, Execution.STOPPED)
    else:
        chosen = current.terminal and result is ResultState.MISSING
    return None if chosen else Hold.SUBMITTED


def _refuse(refused: Mapping[str, list[str]]) -> None:
    reasons = [
        f"{label}: {', '.join(map(repr, refused[name]))}"
        for name, label in _REFUSALS
        if refused[name]
    ]
    keys = tuple(dict.fromkeys(key for name, _ in _REFUSALS for key in refused[name]))
    raise PlanRefused("plan refused; " + "; ".join(reasons), keys=keys)
