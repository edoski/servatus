# pyright: reportPrivateUsage=false
from __future__ import annotations

from datetime import UTC, datetime

from . import _slurm
from ._errors import PlanError
from ._model import (
    AcceptanceState,
    AllocationEvidence,
    AllocationState,
    AttemptEvidence,
    CampaignView,
    ResultProbe,
    ResultState,
    SlurmTarget,
    State,
    TaskEvidence,
)


def observe(
    state: State,
    probe: ResultProbe | None = None,
    *,
    scheduler: bool = True,
    task_keys: tuple[str, ...] | None = None,
) -> CampaignView:
    selected = set(task_keys) if task_keys is not None else {task.key for task in state.tasks}
    tasks = tuple(task for task in state.tasks if task.key in selected)
    relevant = tuple(
        attempt for attempt in state.attempts if selected.intersection(attempt.task_keys)
    )
    results: dict[str, tuple[ResultState, datetime | None]] = {}
    for task in tasks:
        if probe is None:
            results[task.key] = (ResultState.UNOBSERVED, None)
        else:
            valid = probe(task)
            if type(valid) is not bool:
                raise TypeError("result probe must return bool")
            results[task.key] = (
                ResultState.VALID if valid else ResultState.MISSING,
                datetime.now(UTC),
            )

    batches: dict[SlurmTarget, list[_slurm._AttemptQuery]] = {}
    if scheduler:
        for attempt in relevant:
            if attempt.receipt is not None:
                receipt = attempt.receipt
                batches.setdefault(attempt.profile.target, []).append(
                    _slurm._AttemptQuery(
                        attempt.allocation_id,
                        receipt.job_id,
                        receipt.cluster,
                        attempt.window_start,
                        attempt.window_end,
                    )
                )
    observations: dict[str, _slurm.SchedulerObservation] = {}
    for target, queries in batches.items():
        values = _slurm.query_attempts(target, tuple(queries))
        observations.update(
            (query.allocation_id, value) for query, value in zip(queries, values, strict=True)
        )
    now = datetime.now(UTC)
    attempts: list[AttemptEvidence] = []
    current: dict[str, AttemptEvidence] = {}
    ambiguous: set[str] = set()
    for attempt in relevant:
        observation = observations.get(attempt.allocation_id)
        allocation = (
            None
            if observation is None
            else AllocationEvidence(
                observation.state,
                observation.raw_state,
                observation.accounting_state,
                observation.exit_code,
                observation.reason,
                observation.started_at,
                observation.ended_at,
                now,
                observation.retained,
            )
        )
        evidence = AttemptEvidence(
            attempt.allocation_id,
            attempt.task_keys,
            attempt.retry_task_keys,
            attempt.duplicate_risk_task_keys,
            attempt.profile.label,
            attempt.acceptance,
            attempt.receipt,
            allocation,
        )
        attempts.append(evidence)
        if evidence.acceptance is AcceptanceState.ACCEPTED:
            for key in evidence.task_keys:
                if key not in ambiguous:
                    current[key] = evidence
        elif evidence.acceptance is AcceptanceState.UNRESOLVED:
            ambiguous.update(evidence.task_keys)
            for key in evidence.task_keys:
                current[key] = evidence
    task_evidence = tuple(
        TaskEvidence(
            task.key,
            *results[task.key],
            current[task.key].allocation_id if task.key in current else None,
            allocation.state
            if task.key in current and (allocation := current[task.key].allocation) is not None
            else None,
            task.key in ambiguous,
        )
        for task in tasks
    )
    terminal = {AllocationState.SUCCEEDED, AllocationState.FAILED, AllocationState.CANCELLED}
    return CampaignView(
        state.campaign_id,
        state.revision,
        state.sealed,
        task_evidence,
        tuple(attempts),
        scheduler,
        now,
        state.sealed and all(task.result is ResultState.VALID for task in task_evidence),
        scheduler
        and not ambiguous
        and all(
            attempt.allocation is not None
            and attempt.allocation.state in terminal
            and not attempt.allocation.retained
            for attempt in attempts
            if attempt.acceptance is AcceptanceState.ACCEPTED
        ),
    )


def select_tasks(
    view: CampaignView,
    retry: tuple[str, ...],
    duplicate_risk: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    retries = set(retry)
    acknowledged = set(duplicate_risk)
    known = {task.key for task in view.tasks}
    if acknowledged - (retries & known):
        raise PlanError("duplicate-risk acknowledgement requires an explicitly retried Task key")
    active = {AllocationState.QUEUED, AllocationState.RUNNING}
    selected: list[str] = []
    excluded: list[str] = []
    accepted_by_task: dict[str, list[AttemptEvidence]] = {task.key: [] for task in view.tasks}
    for attempt in view.attempts:
        if attempt.acceptance is AcceptanceState.ACCEPTED:
            for key in attempt.task_keys:
                if key in accepted_by_task:
                    accepted_by_task[key].append(attempt)

    for task in view.tasks:
        if task.result is ResultState.VALID:
            if task.key in retries:
                raise PlanError(f"valid task {task.key!r} cannot be retried")
            excluded.append(task.key)
            continue
        if task.acceptance_ambiguous:
            if task.key in retries:
                raise PlanError(f"ambiguous task {task.key!r} cannot be retried")
            excluded.append(task.key)
            continue

        accepted = accepted_by_task[task.key]
        if not accepted:
            if task.key in retries:
                raise PlanError("retry requires an earlier accepted attempt")
            selected.append(task.key)
            continue

        states = tuple(
            attempt.allocation.state for attempt in accepted if attempt.allocation is not None
        )
        if len(states) != len(accepted):
            raise PlanError("accepted attempt lacks scheduler evidence")
        if any(state in active for state in states) or any(
            attempt.allocation is not None and attempt.allocation.retained for attempt in accepted
        ):
            if task.key in retries:
                raise PlanError(f"active task {task.key!r} cannot be retried")
            excluded.append(task.key)
            continue
        if task.key not in retries:
            excluded.append(task.key)
            continue
        if AllocationState.UNKNOWN in states and task.key not in acknowledged:
            raise PlanError(
                f"unknown task {task.key!r} retry requires duplicate-risk acknowledgement"
            )
        selected.append(task.key)

    return tuple(selected), tuple(excluded)
