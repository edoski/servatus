"""Campaign status: a pure, self-contained projection of state, scheduler evidence, and results.

Projection rules:

- The latest accepted Attempt containing a Task owns that Task's current execution. An
  unresolved Attempt (intent recorded, acceptance unknown) dominates instead: the Task is
  ``unresolved`` and its execution is unknown (``None``).
- A Task's execution and exit code follow ``task_execution``: its own ``srun`` step when known,
  else the allocation, except that UNKNOWN or contradictory allocation evidence, and a failed or
  cancelled packed allocation without the Task's step, leave the Task UNKNOWN.
- ``results_ready`` means the roster is sealed and every Task has a VALID result.
- ``quiescent`` means the scheduler was observed, no acceptance is unresolved, and every
  accepted Attempt has terminal, non-retained allocation evidence. Scheduler completion says
  nothing about result validity, and valid results say nothing about stopped work.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import cast

from ..errors import ConfigurationError
from . import _codec
from ._config import task_keys
from ._evidence import AllocationState, JobRef, Observation, SchedulerEvidence, StepEvidence
from ._state import AcceptanceState, State

STATUS_FORMAT = "servatus.status/1"


class ResultState(StrEnum):
    """What the application's result probe said about one Task in this observation."""

    UNOBSERVED = "UNOBSERVED"
    MISSING = "MISSING"
    VALID = "VALID"


@dataclass(frozen=True, slots=True)
class TaskStatus:
    key: str
    result: ResultState
    current_allocation_id: str | None
    execution: AllocationState | None
    exit_code: str | None
    unresolved: bool


@dataclass(frozen=True, slots=True)
class AttemptStatus:
    allocation_id: str
    task_keys: tuple[str, ...]
    retry: tuple[str, ...]
    duplicate_risk: tuple[str, ...]
    profile_label: str
    acceptance: AcceptanceState
    job: JobRef | None
    intent_at: datetime
    scheduler: SchedulerEvidence | None
    steps: tuple[StepEvidence | None, ...]


_UNATTRIBUTED = (AllocationState.FAILED, AllocationState.CANCELLED)
_EXECUTION_COUNTS = ("unsubmitted", "unresolved", "accepted") + tuple(
    state.value.lower() for state in AllocationState
)


@dataclass(frozen=True, slots=True)
class Status:
    """One diagnostic snapshot. It carries no execution authority and no Task payloads."""

    campaign_id: str
    revision: int
    sealed: bool
    observed_at: datetime
    scheduler_observed: bool
    tasks: tuple[TaskStatus, ...]
    attempts: tuple[AttemptStatus, ...]
    results_ready: bool
    quiescent: bool

    def counts(self) -> Mapping[str, int]:
        """Task counts with a fixed set of keys.

        ``tasks`` is the total. ``valid``/``missing``/``unobserved`` partition Tasks by result.
        The execution keys partition them again: ``unsubmitted`` (never accepted),
        ``unresolved``, ``accepted`` (current execution not observed), then one key per
        lowercased ``AllocationState``.
        """
        counts = {"tasks": len(self.tasks)}
        counts.update((state.value.lower(), 0) for state in ResultState)
        counts.update((name, 0) for name in _EXECUTION_COUNTS)
        for task in self.tasks:
            counts[task.result.value.lower()] += 1
            if task.unresolved:
                bucket = "unresolved"
            elif task.current_allocation_id is None:
                bucket = "unsubmitted"
            elif task.execution is None:
                bucket = "accepted"
            else:
                bucket = task.execution.value.lower()
            counts[bucket] += 1
        return counts

    def to_json(self) -> bytes:
        """Canonical JSON tagged ``{"format": "servatus.status/1"}``, including ``counts``.

        Scheduler text and labels may be private; treat the document as private diagnostics.
        """
        document = cast(dict[str, object], _codec.dump(self))
        document["format"] = STATUS_FORMAT
        document["counts"] = dict(self.counts())
        return _codec.canonical(document)


def classify_results(keys: Iterable[str], valid: Collection[str]) -> dict[str, ResultState]:
    """Turn one probe answer for ``keys`` into result states. Unprobed Tasks stay UNOBSERVED."""
    asked = tuple(keys)
    answered = set(task_keys(valid, "a result probe answer"))
    if foreign := answered - set(asked):
        shown = ", ".join(sorted(map(repr, foreign)))
        raise ConfigurationError(f"result probe returned keys it was not asked about: {shown}")
    return {key: ResultState.VALID if key in answered else ResultState.MISSING for key in asked}


def task_execution(
    observation: Observation, slot: int, task_count: int
) -> tuple[AllocationState, str | None]:
    """The Task in ``slot`` of an observed allocation of ``task_count`` Tasks.

    Contradictory (``problem``) or UNKNOWN allocation evidence makes every Task UNKNOWN, whatever
    its step says. Otherwise the Task's own step decides when known. Without one, a single-Task
    allocation is the Task, and so is a successful allocation (its script fails when any step
    does); but a failed or cancelled packed allocation says nothing about one Task, which may
    have finished first, so the Task is UNKNOWN. Exit codes accompany terminal states only.
    """
    allocation = observation.allocation
    if allocation.problem is not None or allocation.state is AllocationState.UNKNOWN:
        return AllocationState.UNKNOWN, None
    step = own_step(observation, slot)
    if step is not None:
        state, exit_code = step.state, step.exit_code
    elif task_count > 1 and allocation.state in _UNATTRIBUTED:
        return AllocationState.UNKNOWN, None
    else:
        state, exit_code = allocation.state, allocation.exit_code
    return state, exit_code if state.terminal else None


def own_step(observation: Observation, slot: int) -> StepEvidence | None:
    """The step evidence of the Task in ``slot``, when the step query found exactly one."""
    return observation.steps[slot] if slot < len(observation.steps) else None


def project(
    state: State,
    observations: Mapping[str, Observation],
    results: Mapping[str, ResultState],
    *,
    scheduler_observed: bool,
    observed_at: datetime,
) -> Status:
    """Project ``state`` with scheduler ``observations`` (by allocation id) and probe
    ``results`` (by Task key; absent keys are UNOBSERVED) into one ``Status``."""
    if not isinstance(observed_at, datetime) or observed_at.utcoffset() != timedelta(0):
        raise ConfigurationError("observed_at must be an aware UTC datetime")
    attempts: list[AttemptStatus] = []
    current: dict[str, tuple[AttemptStatus, int]] = {}
    for attempt in state.attempts:
        accepted = attempt.acceptance is AcceptanceState.ACCEPTED
        observation = observations.get(attempt.allocation_id) if accepted else None
        projected = AttemptStatus(
            allocation_id=attempt.allocation_id,
            task_keys=attempt.task_keys,
            retry=attempt.retry,
            duplicate_risk=attempt.duplicate_risk,
            profile_label=attempt.profile.label,
            acceptance=attempt.acceptance,
            job=attempt.job,
            intent_at=attempt.intent_at,
            scheduler=None if observation is None else observation.allocation,
            steps=() if observation is None else observation.steps,
        )
        attempts.append(projected)
        if attempt.acceptance is not AcceptanceState.NOT_SUBMITTED:
            # The State invariant makes an unresolved Attempt the last one naming its Tasks.
            current.update((key, (projected, slot)) for slot, key in enumerate(attempt.task_keys))
    tasks = tuple(
        _task_status(task.key, current.get(task.key), observations, results) for task in state.tasks
    )
    quiescent = scheduler_observed and all(
        attempt.acceptance is AcceptanceState.NOT_SUBMITTED
        or (
            attempt.scheduler is not None
            and attempt.scheduler.state.terminal
            and not attempt.scheduler.retained
        )
        for attempt in attempts
    )
    return Status(
        campaign_id=state.campaign_id,
        revision=state.revision,
        sealed=state.sealed,
        observed_at=observed_at,
        scheduler_observed=scheduler_observed,
        tasks=tasks,
        attempts=tuple(attempts),
        results_ready=state.sealed and all(task.result is ResultState.VALID for task in tasks),
        quiescent=quiescent,
    )


def _task_status(
    key: str,
    owner: tuple[AttemptStatus, int] | None,
    observations: Mapping[str, Observation],
    results: Mapping[str, ResultState],
) -> TaskStatus:
    """One Task, owned by the Attempt (and slot) that holds its current execution, if any."""
    execution: AllocationState | None = None
    exit_code: str | None = None
    attempt, slot = owner if owner is not None else (None, 0)
    accepted = attempt is not None and attempt.acceptance is AcceptanceState.ACCEPTED
    observation = observations.get(attempt.allocation_id) if attempt and accepted else None
    if attempt is not None and observation is not None:
        execution, exit_code = task_execution(observation, slot, len(attempt.task_keys))
    return TaskStatus(
        key=key,
        result=results.get(key, ResultState.UNOBSERVED),
        current_allocation_id=None if attempt is None else attempt.allocation_id,
        execution=execution,
        exit_code=exit_code,
        unresolved=attempt is not None and attempt.acceptance is AcceptanceState.UNRESOLVED,
    )
