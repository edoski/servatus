"""Campaign status: a pure, self-contained projection of state, scheduler evidence, and results.

Projection rules:

- The latest accepted Attempt containing a Task owns that Task's current execution. An
  unresolved Attempt (intent recorded, acceptance unknown) dominates instead: the Task is
  ``unresolved`` and its execution is unknown (``None``).
- A Task's execution and exit code follow ``Observation.execution`` for its slot: its queue row
  while Slurm lists the job, then its run markers.
- ``results_ready`` means the roster is sealed and every Task has a VALID result.
- ``quiescent`` means the scheduler was observed, no acceptance is unresolved, and Slurm lists no
  job of any accepted allocation, nor of one recorded as not submitted. Scheduler completion says
  nothing about result validity, and valid results say nothing about stopped work.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import cast

from ..errors import ConfigurationError
from . import _codec
from ._config import task_keys
from ._evidence import Execution, Observation, Observed
from ._state import AcceptanceState, State

STATUS_FORMAT = "servatus.status/1"
NEVER_STARTED = "never started"


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
    execution: Execution | None
    exit_code: int | None
    detail: str | None
    unresolved: bool


@dataclass(frozen=True, slots=True)
class AttemptStatus:
    allocation_id: str
    task_keys: tuple[str, ...]
    retry: tuple[str, ...]
    profile_label: str
    acceptance: AcceptanceState
    job: int | None
    intent_at: datetime
    observation: Observation | None


_EXECUTION_COUNTS = ("unsubmitted", "unresolved", "accepted") + tuple(
    state.value.lower() for state in Execution
)


@dataclass(frozen=True, slots=True)
class Status:
    """One diagnostic snapshot. It carries no execution authority and no Task payloads.

    ``warnings`` name allocations recorded as not submitted for which Slurm still lists a job;
    their Tasks are held ``ACTIVE`` until it leaves the queue.
    """

    campaign_id: str
    revision: int
    sealed: bool
    observed_at: datetime
    scheduler_observed: bool
    tasks: tuple[TaskStatus, ...]
    attempts: tuple[AttemptStatus, ...]
    results_ready: bool
    quiescent: bool
    warnings: tuple[str, ...]

    def counts(self) -> Mapping[str, int]:
        """Task counts with a fixed set of keys.

        ``tasks`` is the total. ``valid``/``missing``/``unobserved`` partition Tasks by result.
        The execution keys partition them again: ``unsubmitted`` (never accepted),
        ``unresolved``, ``accepted`` (current execution not observed), then one key per
        lowercased ``Execution``.
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


def stray_warning(allocation_id: str, job: int) -> str:
    return (
        f"allocation {allocation_id} is recorded as not submitted, but Slurm lists job {job} "
        "for it; its Tasks stay held ACTIVE until it leaves the queue"
    )


def project(
    state: State,
    observed: Observed,
    results: Mapping[str, ResultState],
    *,
    scheduler_observed: bool,
    observed_at: datetime,
) -> Status:
    """Project ``state`` with scheduler evidence and probe ``results`` (by Task key; absent keys
    are UNOBSERVED) into one ``Status``."""
    attempts: list[AttemptStatus] = []
    current: dict[str, tuple[AttemptStatus, int]] = {}
    accepted: set[str] = set()
    for attempt in state.attempts:
        projected = AttemptStatus(
            allocation_id=attempt.allocation_id,
            task_keys=attempt.task_keys,
            retry=tuple(key for key in attempt.task_keys if key in accepted),
            profile_label=attempt.profile.label,
            acceptance=attempt.acceptance,
            job=attempt.job,
            intent_at=attempt.intent_at,
            observation=(
                None if attempt.job is None else observed.attempts.get(attempt.allocation_id)
            ),
        )
        attempts.append(projected)
        if attempt.acceptance is AcceptanceState.ACCEPTED:
            accepted.update(attempt.task_keys)
        if attempt.acceptance is not AcceptanceState.NOT_SUBMITTED:
            # The State invariant makes an unresolved Attempt the last one naming its Tasks.
            current.update((key, (projected, slot)) for slot, key in enumerate(attempt.task_keys))
    tasks = tuple(_task_status(task.key, current.get(task.key), results) for task in state.tasks)
    quiescent = (
        scheduler_observed
        and not observed.strays
        and all(
            attempt.acceptance is AcceptanceState.NOT_SUBMITTED
            or (attempt.observation is not None and not attempt.observation.live)
            for attempt in attempts
        )
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
        warnings=tuple(stray_warning(name, job) for name, job in observed.strays.items()),
    )


def _task_status(
    key: str, owner: tuple[AttemptStatus, int] | None, results: Mapping[str, ResultState]
) -> TaskStatus:
    """One Task, owned by the Attempt (and slot) that holds its current execution, if any."""
    attempt, slot = owner if owner is not None else (None, 0)
    observation = None if attempt is None else attempt.observation
    execution: Execution | None = None
    exit_code: int | None = None
    detail: str | None = None
    if observation is not None:
        execution, exit_code = observation.execution(slot)
        if execution is Execution.STOPPED and not observation.started:
            detail = NEVER_STARTED
    return TaskStatus(
        key=key,
        result=results.get(key, ResultState.UNOBSERVED),
        current_allocation_id=None if attempt is None else attempt.allocation_id,
        execution=execution,
        exit_code=exit_code,
        detail=detail,
        unresolved=attempt is not None and attempt.acceptance is AcceptanceState.UNRESOLVED,
    )
