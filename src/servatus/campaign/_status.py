"""Campaign status: a pure, self-contained projection of state, scheduler evidence, and results.

Projection rules:

- The latest accepted Attempt containing a Task owns that Task's current execution. An
  unresolved Attempt (intent recorded, acceptance unknown) dominates instead: the Task is
  ``unresolved`` and its execution is unknown (``None``).
- A Task's execution and exit code come from its own ``srun`` step when the observation has
  that step, and from the allocation otherwise.
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
    if isinstance(valid, (str, bytes)) or not isinstance(valid, Collection):
        raise ConfigurationError("a result probe must return a collection of Task keys")
    answered = set(valid)
    if foreign := answered - set(asked):
        shown = ", ".join(sorted(map(repr, foreign)))
        raise ConfigurationError(f"result probe returned keys it was not asked about: {shown}")
    return {key: ResultState.VALID if key in answered else ResultState.MISSING for key in asked}


def task_execution(observation: Observation, slot: int) -> tuple[AllocationState, str | None]:
    """The Task in ``slot`` of an observed allocation: its step if known, else the allocation."""
    step = observation.steps[slot] if slot < len(observation.steps) else None
    if step is not None:
        return step.state, step.exit_code
    return observation.allocation.state, observation.allocation.exit_code


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
    unresolved: set[str] = set()
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
        if attempt.acceptance is AcceptanceState.UNRESOLVED:
            unresolved.update(attempt.task_keys)
        if attempt.acceptance is not AcceptanceState.NOT_SUBMITTED:
            # The State invariant makes an unresolved Attempt the last one naming its Tasks.
            for slot, key in enumerate(attempt.task_keys):
                current[key] = (projected, slot)
    tasks: list[TaskStatus] = []
    for task in state.tasks:
        owner = current.get(task.key)
        execution: AllocationState | None = None
        exit_code: str | None = None
        if owner is not None and owner[0].acceptance is AcceptanceState.ACCEPTED:
            observation = observations.get(owner[0].allocation_id)
            if observation is not None:
                execution, exit_code = task_execution(observation, owner[1])
        tasks.append(
            TaskStatus(
                key=task.key,
                result=results.get(task.key, ResultState.UNOBSERVED),
                current_allocation_id=None if owner is None else owner[0].allocation_id,
                execution=execution,
                exit_code=exit_code,
                unresolved=task.key in unresolved,
            )
        )
    quiescent = (
        scheduler_observed
        and not unresolved
        and all(
            attempt.scheduler is not None
            and attempt.scheduler.state.terminal
            and not attempt.scheduler.retained
            for attempt in attempts
            if attempt.acceptance is AcceptanceState.ACCEPTED
        )
    )
    return Status(
        campaign_id=state.campaign_id,
        revision=state.revision,
        sealed=state.sealed,
        observed_at=observed_at,
        scheduler_observed=scheduler_observed,
        tasks=tuple(tasks),
        attempts=tuple(attempts),
        results_ready=state.sealed and all(task.result is ResultState.VALID for task in tasks),
        quiescent=quiescent,
    )
