"""Campaign status: a pure, self-contained projection of state, scheduler evidence, and results.

Projection rules:

- A Task's unresolved Attempt (intent recorded, acceptance unknown) dominates: its execution is
  ``UNRESOLVED``. Otherwise its ``current`` allocation owns its execution, which follows
  ``Observation.execution`` for its slot (its queue row while Slurm lists the job, then its run
  markers), or is ``ACCEPTED`` when the scheduler was not observed. A Task never accepted is
  ``NOT_SUBMITTED``.
- A Task's run time is the sum, over every allocation that ran it, of the time between run-marker
  mtimes (``Observation.run_time``).
- ``results_ready`` means the roster is sealed and every Task has a VALID result.
- ``quiescent`` means the scheduler was observed, no acceptance is unresolved, and Slurm lists no
  job of any accepted allocation, nor of one recorded as not submitted. Scheduler completion says
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
from ._evidence import Execution, Observation, Observed, StopReason
from ._state import AcceptanceState, Attempt, Kind, Link, State, TaskHistory, history

STATUS_FORMAT = "servatus.status/4"
_SECOND = timedelta(seconds=1)


class ResultState(StrEnum):
    """What the application's result probe said about one Task in this observation."""

    UNOBSERVED = "UNOBSERVED"
    MISSING = "MISSING"
    VALID = "VALID"


@dataclass(frozen=True, slots=True)
class RunTime:
    """A Task's cumulative run time, in whole seconds, over every allocation that ran it.
    ``complete`` is false while it runs, and when a LOST execution counts at its allocation's
    time limit."""

    total: timedelta
    complete: bool


@dataclass(frozen=True, slots=True)
class TaskStatus:
    """One Task: its observed ``Execution``, or the ``AcceptanceState`` of its current Attempt when
    there is no execution to show (``NOT_SUBMITTED`` when it has none).

    ``detail`` says why a ``STOPPED`` Task stopped when the evidence says. ``run_time`` is
    ``None`` when the scheduler was not observed. ``link`` is the current Attempt's place in its
    chain."""

    key: str
    result: ResultState
    current_allocation_id: str | None
    execution: Execution | AcceptanceState
    exit_code: int | None
    detail: StopReason | None
    run_time: RunTime | None
    link: Link | None


@dataclass(frozen=True, slots=True)
class AttemptStatus:
    """One Attempt: an allocation, a lease of ``slots``, or a step using ``slots`` of the lease
    it runs ``within`` (``kind``)."""

    allocation_id: str
    link: Link
    task_keys: tuple[str, ...]
    retry: tuple[str, ...]
    profile_label: str
    acceptance: AcceptanceState
    job: int | None
    intent_at: datetime
    observation: Observation | None
    slots: int | None
    within: str | None

    @property
    def kind(self) -> Kind:
        return Kind.of(self.slots, self.within)


_EXECUTIONS = (
    AcceptanceState.NOT_SUBMITTED,
    AcceptanceState.UNRESOLVED,
    AcceptanceState.ACCEPTED,
    *Execution,
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

        ``tasks`` is the total. ``valid``/``missing``/``unobserved`` partition Tasks by result,
        and one lowercased key per ``TaskStatus.execution`` value partitions them again:
        ``not_submitted``, ``unresolved``, ``accepted`` (not observed), then each ``Execution``.
        """
        counts = {"tasks": len(self.tasks)}
        counts.update((state.value.lower(), 0) for state in ResultState)
        counts.update((state.value.lower(), 0) for state in _EXECUTIONS)
        for task in self.tasks:
            counts[task.result.value.lower()] += 1
            counts[task.execution.value.lower()] += 1
        return counts

    def to_json(self) -> bytes:
        """Canonical JSON tagged ``{"format": "servatus.status/4"}``, including ``counts``.

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


def current(past: TaskHistory, observed: Observed) -> tuple[Attempt, int] | None:
    """The Task's current allocation and its slot there: the latest accepted Attempt that was
    handed the Task. Link 1 of a chain always is; a later link is once its observation shows the
    handoff. Without observations, the latest accepted link 1."""
    found: tuple[Attempt, int] | None = None
    for attempt, slot in past.accepted:
        observation = observed.attempts.get(attempt.allocation_id)
        if attempt.link.index == 1 or (observation is not None and observation.handed[slot]):
            found = attempt, slot
    return found


def run_time(past: TaskHistory, observed: Observed, now: datetime) -> RunTime | None:
    """The Task's run time over every accepted Attempt, or ``None`` when one was not observed."""
    total, complete = timedelta(0), True
    for attempt, slot in past.accepted:
        observation = observed.attempts.get(attempt.allocation_id)
        if observation is None:
            return None
        spent, exact = observation.run_time(slot, now, attempt.profile.resources.time_limit)
        total += spent
        complete = complete and exact
    return RunTime(total - total % _SECOND, complete)


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
    accepted: set[str] = set()
    for attempt in state.attempts:
        projected = AttemptStatus(
            allocation_id=attempt.allocation_id,
            link=attempt.link,
            task_keys=attempt.task_keys,
            retry=tuple(key for key in attempt.task_keys if key in accepted),
            profile_label=attempt.profile.label,
            acceptance=attempt.acceptance,
            job=attempt.job,
            intent_at=attempt.intent_at,
            observation=(
                None if attempt.job is None else observed.attempts.get(attempt.allocation_id)
            ),
            slots=attempt.slots,
            within=attempt.within,
        )
        attempts.append(projected)
        if attempt.acceptance is AcceptanceState.ACCEPTED:
            accepted.update(attempt.task_keys)
    tasks = tuple(
        _task_status(
            key,
            past,
            observed,
            results.get(key, ResultState.UNOBSERVED),
            observed_at if scheduler_observed else None,
        )
        for key, past in history(state).items()
    )
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
    key: str, past: TaskHistory, observed: Observed, result: ResultState, now: datetime | None
) -> TaskStatus:
    """``now`` is ``None`` when the scheduler was not observed, so run time is unknown."""
    owner: Attempt | None = past.unresolved
    execution: Execution | AcceptanceState = AcceptanceState.UNRESOLVED
    exit_code: int | None = None
    detail: StopReason | None = None
    if owner is None and (found := current(past, observed)) is not None:
        owner, slot = found
        observation = observed.attempts.get(owner.allocation_id)
        if observation is None:
            execution = AcceptanceState.ACCEPTED
        else:
            execution, exit_code, detail = observation.execution(slot)
            last = owner.link.budget is not None and owner.link.index == owner.link.count
            if last and detail is StopReason.YIELDED:
                detail = StopReason.BUDGET_SPENT
    elif owner is None:
        execution = AcceptanceState.NOT_SUBMITTED
    return TaskStatus(
        key=key,
        result=result,
        current_allocation_id=None if owner is None else owner.allocation_id,
        execution=execution,
        exit_code=exit_code,
        detail=detail,
        run_time=None if now is None else run_time(past, observed, now),
        link=None if owner is None else owner.link,
    )
