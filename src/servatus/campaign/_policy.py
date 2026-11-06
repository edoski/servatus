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
from ._state import State, TaskHistory, history
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


class _Refusal(Enum):
    """Why an explicitly named key refuses the whole plan, in reporting order."""

    UNKNOWN = "unknown Task keys"
    VALID = "valid results cannot be retried"
    UNACCEPTED = "retry needs an earlier accepted attempt"
    LIVE = "Slurm still lists earlier work; wait for it or cancel it"


@dataclass(frozen=True, slots=True)
class Request:
    """One plan request, parsed once: explicit retry keys or a bulk selector, and an optional
    ``only`` scope."""

    retry: frozenset[str] | Retry = frozenset()
    only: frozenset[str] | None = None

    @classmethod
    def parse(
        cls, retry: Collection[str] | Retry = (), only: Collection[str] | None = None
    ) -> Request:
        return cls(
            retry if isinstance(retry, Retry) else frozenset(task_keys(retry, "retry")),
            None if only is None else frozenset(task_keys(only, "only")),
        )

    def considers(self, key: str, past: TaskHistory) -> bool:
        """Whether ``key`` could be selected: in scope, and fresh or open to retry."""
        return (self.only is None or key in self.only) and (
            not past.accepted or isinstance(self.retry, Retry) or key in self.retry
        )


@dataclass(frozen=True, slots=True)
class Selection:
    """Selected Task keys in roster order and the reason for every other Task."""

    selected: tuple[str, ...]
    held: Mapping[str, Hold]


def scope(state: State, request: Request) -> tuple[tuple[Task, ...], tuple[str, ...]]:
    """What ``decide`` may consult for ``request``: the Tasks to probe, and the allocation ids
    to observe, in state order.

    Only Tasks that could be selected count. Their accepted and not-submitted Attempts are
    observed unless one is unresolved, which holds the Task anyway, so a plan of fresh work
    without such history observes nothing. Unknown keys are ignored here; ``decide`` refuses
    them.
    """
    past = history(state)
    tasks = tuple(task for task in state.tasks if request.considers(task.key, past[task.key]))
    wanted: set[str] = set()
    for task in tasks:
        if (own := past[task.key]).unresolved is None:
            wanted.update(attempt.allocation_id for attempt, _ in own.accepted)
            wanted.update(attempt.allocation_id for attempt in own.not_submitted)
    observed = tuple(item.allocation_id for item in state.attempts if item.allocation_id in wanted)
    return tasks, observed


def decide(
    state: State, observed: Observed, results: Mapping[str, ResultState], request: Request
) -> Selection:
    """Apply the eligibility policy in the module docstring to every Task in the roster.

    ``observed`` must cover ``scope``; ``results`` maps Task keys to probe results, with absent
    keys UNOBSERVED.
    """
    past = history(state)
    refused: dict[_Refusal, list[str]] = {reason: [] for reason in _Refusal}
    named: frozenset[str] = frozenset() if isinstance(request.retry, Retry) else request.retry
    refused[_Refusal.UNKNOWN] = sorted((named | (request.only or frozenset())) - past.keys())
    selected: list[str] = []
    held: dict[str, Hold] = {}
    for key, own in past.items():
        if request.only is not None and key not in request.only:
            outcome: Hold | _Refusal | None = Hold.NOT_REQUESTED
        else:
            result = results.get(key, ResultState.UNOBSERVED)
            outcome = _judge(own, observed, result, request.retry, named=key in named)
        if outcome is None:
            selected.append(key)
        elif isinstance(outcome, Hold):
            held[key] = outcome
        else:
            refused[outcome].append(key)
    if any(refused.values()):
        reasons = [
            f"{reason.value}: {', '.join(map(repr, keys))}"
            for reason, keys in refused.items()
            if keys
        ]
        keys = tuple(dict.fromkeys(key for keys in refused.values() for key in keys))
        raise PlanRefused("plan refused; " + "; ".join(reasons), keys=keys)
    return Selection(tuple(selected), MappingProxyType(held))


def _judge(
    past: TaskHistory,
    observed: Observed,
    result: ResultState,
    retry: frozenset[str] | Retry,
    *,
    named: bool,
) -> Hold | _Refusal | None:
    """One requested Task: its hold reason, a refusal (named keys only), or None to select it."""
    if result is ResultState.VALID:
        return _Refusal.VALID if named else Hold.VALID
    if past.unresolved is not None:
        return Hold.UNRESOLVED
    stray = any(attempt.allocation_id in observed.strays for attempt in past.not_submitted)
    if not past.accepted:
        if named:
            return _Refusal.UNACCEPTED
        return Hold.ACTIVE if stray else None
    if not named and not isinstance(retry, Retry):
        return Hold.SUBMITTED
    if stray or any(observed.attempts[attempt.allocation_id].live for attempt, _ in past.accepted):
        return _Refusal.LIVE if named else Hold.ACTIVE
    if named:
        return None
    attempt, slot = past.accepted[-1]
    current, _ = observed.attempts[attempt.allocation_id].execution(slot)
    if retry is Retry.FAILED:
        chosen = current in (Execution.FAILED, Execution.STOPPED)
    else:
        chosen = current.terminal and result is ResultState.MISSING
    return None if chosen else Hold.SUBMITTED
