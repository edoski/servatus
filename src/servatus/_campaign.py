# pyright: reportPrivateUsage=false, reportUnnecessaryIsInstance=false
from __future__ import annotations

import os
from collections.abc import Collection, Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from . import _slurm
from ._errors import (
    ConfigurationError,
    ObservationError,
    PlanError,
    ReconciliationError,
    SubmissionError,
    TaskConflict,
)
from ._model import (
    SCHEMA_VERSION,
    AcceptanceState,
    Attempt,
    CampaignView,
    JobReceipt,
    LogSnapshot,
    PlannedAllocation,
    Profile,
    RegisteredTask,
    ResourceRequest,
    ResultProbe,
    SlurmTarget,
    State,
    SubmissionPlan,
    SubmitResult,
    Task,
    UnresolvedSubmission,
    ValidationResult,
    _duration_seconds,
    _integer,
    _safe_token,
    boolean,
    canonical,
    digest,
    effective_time_limit,
    identifier,
    integer,
    keys,
    object_fields,
    profile_document,
    profile_from_document,
    receipt_document,
)
from ._observation import observe, select_tasks
from ._store import Store


class Campaign:
    def __init__(self, store: Store) -> None:
        self._store = store

    @classmethod
    def create(cls, path: Path, tasks: Sequence[Task], *, appendable: bool = False) -> Self:
        frozen = _tasks(tasks)
        if type(appendable) is not bool:
            raise ConfigurationError("appendable must be bool")
        store = Store.create(path)
        with store.transaction(creating=True) as tx:
            if tx.state is not None:
                raise TaskConflict("campaign already exists; use load")
            tx.commit(
                State(
                    os.urandom(16).hex(),
                    0,
                    tuple(RegisteredTask(task, 0) for task in frozen),
                    None if appendable else 0,
                )
            )
        return cls(store)

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls(Store.load(path))

    @property
    def tasks(self) -> tuple[Task, ...]:
        return self._store.read().tasks

    def append(self, tasks: Sequence[Task]) -> None:
        suffix = _tasks(tasks)
        with self._store.transaction() as tx:
            state = tx.state
            assert state is not None
            if state.sealed:
                raise TaskConflict("sealed campaign tasks cannot change")
            if {task.key for task in suffix}.intersection(task.key for task in state.tasks):
                raise TaskConflict("appended Task keys must be new")
            if suffix:
                revision = state.revision + 1
                tx.commit(
                    replace(
                        state,
                        revision=revision,
                        roster=state.roster
                        + tuple(RegisteredTask(task, revision) for task in suffix),
                    )
                )

    def seal(self) -> None:
        with self._store.transaction() as tx:
            state = tx.state
            assert state is not None
            if not state.sealed:
                tx.commit(
                    replace(state, revision=state.revision + 1, sealed_revision=state.revision + 1)
                )

    def inspect(self, probe: ResultProbe | None = None, *, scheduler: bool = True) -> CampaignView:
        state = self._store.read()
        view = observe(state, probe, scheduler=scheduler)
        if (
            after := self._store.read()
        ).campaign_id != state.campaign_id or after.revision != state.revision:
            raise ObservationError("campaign changed during inspection")
        return view

    def plan(
        self,
        profile: Profile,
        probe: ResultProbe | None = None,
        *,
        retry: Collection[str] = (),
        allow_duplicate_risk: Collection[str] = (),
        tasks_per_allocation: int | None = None,
    ) -> SubmissionPlan:
        state = self._store.read()
        known = tuple(task.key for task in state.tasks)
        retries = _selection(retry, known, "retry")
        ack = _selection(allow_duplicate_risk, known, "duplicate risk")
        view = observe(state, probe)
        if (
            after := self._store.read()
        ).campaign_id != state.campaign_id or after.revision != state.revision:
            raise PlanError("campaign changed during planning")
        eligible, excluded = select_tasks(view, retries, ack)
        capacity = _capacity(profile.target, profile.resources)
        if tasks_per_allocation is not None:
            _integer(tasks_per_allocation, minimum=1, name="tasks_per_allocation")
            if tasks_per_allocation > capacity:
                raise PlanError("tasks_per_allocation exceeds feasible capacity")
            capacity = tasks_per_allocation
        bound = capacity * profile.target.max_allocations_per_submit
        selected, deferred = eligible[:bound], eligible[bound:]
        return _build_plan(
            state,
            profile,
            selected,
            excluded,
            deferred,
            tuple(key for key in retries if key in selected),
            tuple(key for key in ack if key in selected),
            capacity,
            probe is not None,
        )

    def submit(self, plan: SubmissionPlan, *, probe: ResultProbe | None = None) -> SubmitResult:
        self._preflight_plan(plan)
        if plan.probe_required and probe is None and plan.allocations:
            raise PlanError("result-aware submission requires a result probe")
        expected_revision = plan.revision
        receipts: list[JobReceipt] = []
        for index, allocation in enumerate(plan.allocations):
            try:
                state = self._store.read()
                if state.campaign_id != plan.campaign_id or state.revision != expected_revision:
                    raise PlanError("campaign changed before submission freshness check")
                view = observe(
                    state, probe if plan.probe_required else None, task_keys=allocation.task_keys
                )
                retries = tuple(key for key in plan.retry_task_keys if key in allocation.task_keys)
                ack = tuple(
                    key for key in plan.duplicate_risk_task_keys if key in allocation.task_keys
                )
                eligible, _ = select_tasks(view, retries, ack)
                if eligible != allocation.task_keys:
                    raise PlanError("selected tasks became ineligible before submission")
                with self._store.transaction() as tx:
                    current = tx.state
                    assert current is not None
                    if (
                        current.campaign_id != plan.campaign_id
                        or current.revision != expected_revision
                    ):
                        raise PlanError("campaign changed during submission freshness check")
                    attempt = Attempt(
                        allocation.allocation_id,
                        allocation.task_keys,
                        plan.profile,
                        retries,
                        ack,
                        current.revision + 1,
                        datetime.now(UTC),
                    )
                    # Commit can fail after replacement. Recovery must inspect this intent.
                    try:
                        tx.commit(
                            replace(
                                current,
                                revision=attempt.intent_revision,
                                attempts=current.attempts + (attempt,),
                            )
                        )
                    except Exception:
                        return SubmitResult(
                            tuple(receipts),
                            (UnresolvedSubmission(allocation.allocation_id, allocation.task_keys),),
                            plan.allocations[index + 1 :],
                            "intent persistence failed; inspect campaign",
                        )
            except Exception as error:
                return SubmitResult(
                    tuple(receipts),
                    (),
                    plan.allocations[index:],
                    f"submission stopped before next intent: {type(error).__name__}",
                )
            try:
                result = _slurm._run_ssh(plan.profile.target, allocation.argv, allocation.script)
                if result.returncode != 0:
                    raise SubmissionError("no acceptance receipt")
                job_id, cluster = _slurm.parse_receipt(result.stdout)
                receipt = JobReceipt(
                    allocation.allocation_id, job_id, cluster, allocation.task_keys
                )
            except Exception:
                return SubmitResult(
                    tuple(receipts),
                    (UnresolvedSubmission(allocation.allocation_id, allocation.task_keys),),
                    plan.allocations[index + 1 :],
                    "scheduler acceptance is unresolved",
                )
            try:
                revision, outcome_revision = self._complete(attempt.allocation_id, receipt)
            except Exception:
                return SubmitResult(
                    tuple(receipts),
                    (
                        UnresolvedSubmission(
                            allocation.allocation_id, allocation.task_keys, receipt
                        ),
                    ),
                    plan.allocations[index + 1 :],
                    "receipt persistence failed; acceptance observed",
                )
            receipts.append(receipt)
            expected_revision = attempt.intent_revision + 1
            if index + 1 < len(plan.allocations) and (
                revision != expected_revision or outcome_revision != expected_revision
            ):
                return SubmitResult(
                    tuple(receipts),
                    (),
                    plan.allocations[index + 1 :],
                    "campaign changed during submission",
                )
        return SubmitResult(tuple(receipts), (), (), None)

    def _preflight_plan(self, plan: SubmissionPlan, *, validation: bool = False) -> None:
        state = self._store.read()
        if state.campaign_id != plan.campaign_id or state.revision != plan.revision:
            raise PlanError("plan is stale or foreign")
        for allocation in plan.allocations:
            argv = (*allocation.argv, "--test-only") if validation else allocation.argv
            try:
                _slurm.preflight(plan.profile.target, argv)
            except ObservationError as error:
                raise PlanError(str(error)) from error

    def validate(self, plan: SubmissionPlan) -> tuple[ValidationResult, ...]:
        self._preflight_plan(plan, validation=True)
        seen: set[int] = set()
        results: list[ValidationResult] = []
        for allocation in plan.allocations:
            count = len(allocation.task_keys)
            if count in seen:
                continue
            seen.add(count)
            result = _slurm._run_ssh(
                plan.profile.target, (*allocation.argv, "--test-only"), allocation.script
            )
            if result.returncode != 0:
                raise SubmissionError("Slurm rejected time-specific validation")
            results.append(
                ValidationResult(
                    count,
                    allocation.cpus,
                    allocation.memory_mib,
                    allocation.gpus,
                    allocation.time_limit,
                    result.stdout.decode("utf-8", "replace").rstrip("\n"),
                    result.stderr.decode("utf-8", "replace").rstrip("\n"),
                )
            )
        return tuple(results)

    def reconcile(self, allocation_id: str) -> JobReceipt:
        attempt = _attempt(self._store.read(), allocation_id)
        if attempt.acceptance is not AcceptanceState.UNRESOLVED:
            raise ReconciliationError("allocation is not unresolved")
        match = _slurm.query_identity(
            attempt.profile.target,
            job_name=f"servatus-{allocation_id}",
            window_start=attempt.window_start,
            window_end=attempt.window_end,
        )
        receipt = JobReceipt(allocation_id, match.job_id, match.cluster, attempt.task_keys)
        self._complete(allocation_id, receipt)
        return receipt

    def resolve(
        self, allocation_id: str, *, job_id: int | None, cluster: str | None = None
    ) -> None:
        if job_id is not None:
            _integer(job_id, minimum=1, name="job_id")
        _safe_token(cluster, name="cluster", optional=True)
        if job_id is None and cluster is not None:
            raise ConfigurationError("cluster requires a job_id")
        attempt = _attempt(self._store.read(), allocation_id)
        receipt = (
            None
            if job_id is None
            else JobReceipt(allocation_id, job_id, cluster, attempt.task_keys)
        )
        self._complete(allocation_id, receipt)

    def _complete(self, allocation_id: str, receipt: JobReceipt | None) -> tuple[int, int]:
        with self._store.transaction() as tx:
            state = tx.state
            assert state is not None
            attempt = _attempt(state, allocation_id)
            acceptance = (
                AcceptanceState.NOT_SUBMITTED if receipt is None else AcceptanceState.ACCEPTED
            )
            if receipt is not None and receipt.task_keys != attempt.task_keys:
                raise ReconciliationError("receipt differs from allocation intent")
            if attempt.acceptance is not AcceptanceState.UNRESOLVED:
                if attempt.acceptance is acceptance and attempt.receipt == receipt:
                    assert attempt.outcome_revision is not None
                    return state.revision, attempt.outcome_revision
                raise ReconciliationError("allocation already has a conflicting outcome")
            revision = state.revision + 1
            completed = replace(
                attempt, acceptance=acceptance, receipt=receipt, outcome_revision=revision
            )
            tx.commit(
                replace(
                    state,
                    revision=revision,
                    attempts=tuple(
                        completed if item.allocation_id == allocation_id else item
                        for item in state.attempts
                    ),
                )
            )
            return revision, revision

    def read_log(
        self, allocation_id: str, *, task_key: str | None = None, max_bytes: int = 65_536
    ) -> LogSnapshot:
        if type(max_bytes) is not int or not 1 <= max_bytes <= 1024 * 1024:
            raise ConfigurationError("max_bytes must be an integer between 1 and 1048576")
        state = self._store.read()
        attempt = next(
            (item for item in state.attempts if item.allocation_id == allocation_id), None
        )
        if (
            attempt is None
            or attempt.receipt is None
            or (task_key is not None and task_key not in attempt.task_keys)
        ):
            raise ObservationError("campaign log is unavailable")
        slot = None if task_key is None else attempt.task_keys.index(task_key)
        target = attempt.profile.target
        path = _slurm._log_path(target.log_root, allocation_id, attempt.receipt.job_id, slot)
        content, truncated = _slurm.read_log_suffix(target, path, max_bytes)
        return LogSnapshot(content, truncated, datetime.now(UTC))

    def record(self, view: CampaignView) -> bytes:
        state = self._store.read()
        if view.campaign_id != state.campaign_id or view.revision != state.revision:
            raise ObservationError("operational record view is stale or foreign")
        observations = {item.allocation_id: item.allocation for item in view.attempts}
        return canonical(
            {
                "campaign_id": state.campaign_id,
                "revision": state.revision,
                "sealed": state.sealed,
                "task_keys": [task.key for task in state.tasks],
                "attempts": [
                    {
                        "allocation_id": item.allocation_id,
                        "task_keys": list(item.task_keys),
                        "profile_label": item.profile.label,
                        "allocation": {
                            "task_count": len(item.task_keys),
                            "cpus": len(item.task_keys) * item.profile.resources.cpus_per_task,
                            "memory_mib": len(item.task_keys)
                            * item.profile.resources.memory_mib_per_task,
                            "gpus": len(item.task_keys) * item.profile.resources.gpus_per_task,
                            "time_limit": effective_time_limit(item.profile.resources.time_limit),
                        },
                        "retry_task_keys": list(item.retry_task_keys),
                        "duplicate_risk_task_keys": list(item.duplicate_risk_task_keys),
                        "intent_revision": item.intent_revision,
                        "outcome_revision": item.outcome_revision,
                        "acceptance": item.acceptance.value,
                        "receipt": None if item.receipt is None else receipt_document(item.receipt),
                        "execution": observation.state.value
                        if (observation := observations.get(item.allocation_id)) is not None
                        else None,
                        "retained": observation.retained
                        if (observation := observations.get(item.allocation_id)) is not None
                        else None,
                    }
                    for item in state.attempts
                ],
            }
        )


def _tasks(tasks: Sequence[Task]) -> tuple[Task, ...]:
    frozen = tuple(tasks)
    if any(not isinstance(task, Task) for task in frozen):
        raise ConfigurationError("tasks must contain Task values")
    if len({task.key for task in frozen}) != len(frozen):
        raise ConfigurationError("Task keys must be unique")
    return frozen


def _selection(values: Collection[str], known: tuple[str, ...], name: str) -> tuple[str, ...]:
    if isinstance(values, str) or any(not isinstance(key, str) for key in values):
        raise PlanError(f"{name} must contain Task keys")
    if len(set(values)) != len(values) or set(values) - set(known):
        raise PlanError(f"{name} contains duplicate or unknown Task keys")
    return tuple(key for key in known if key in values)


def _attempt(state: State, allocation_id: str) -> Attempt:
    for attempt in state.attempts:
        if attempt.allocation_id == allocation_id:
            return attempt
    raise ReconciliationError("unknown allocation identity")


def _decision(plan: SubmissionPlan) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": plan.campaign_id,
        "revision": plan.revision,
        "profile": profile_document(plan.profile),
        "selected_task_keys": list(plan.selected_task_keys),
        "excluded_task_keys": list(plan.excluded_task_keys),
        "deferred_task_keys": list(plan.deferred_task_keys),
        "retry_task_keys": list(plan.retry_task_keys),
        "duplicate_risk_task_keys": list(plan.duplicate_risk_task_keys),
        "tasks_per_allocation": plan.tasks_per_allocation,
        "probe_required": plan.probe_required,
    }


def plan_document(plan: SubmissionPlan) -> dict[str, object]:
    return {**_decision(plan), "digest": plan.digest}


def _build_plan(
    state: State,
    profile: Profile,
    selected: tuple[str, ...],
    excluded: tuple[str, ...],
    deferred: tuple[str, ...],
    retry: tuple[str, ...],
    ack: tuple[str, ...],
    capacity: int,
    probe_required: bool,
) -> SubmissionPlan:
    if capacity > _capacity(profile.target, profile.resources):
        raise PlanError("tasks_per_allocation exceeds feasible capacity")
    if len(selected) > capacity * profile.target.max_allocations_per_submit:
        raise PlanError("plan exceeds allocation batch limit")
    plan = SubmissionPlan(
        state.campaign_id,
        state.revision,
        profile,
        selected,
        excluded,
        deferred,
        retry,
        ack,
        capacity,
        probe_required,
        (),
        "",
    )
    decision = _decision(plan)
    by_key = {task.key: task for task in state.tasks}
    groups = _balanced_groups(tuple(by_key[key] for key in selected), capacity)
    allocations: list[PlannedAllocation] = []
    for index, group in enumerate(groups):
        allocation_id = digest({"decision": decision, "index": index})[:24]
        script = _slurm.render_script(profile.target, profile.resources, group, allocation_id)
        if len(script) > profile.target.max_script_bytes:
            raise PlanError("rendered script exceeds target max_script_bytes")
        time_limit = effective_time_limit(profile.resources.time_limit)
        argv = _slurm.sbatch_argv(
            profile.target, profile.resources, len(group), allocation_id, time_limit
        )
        try:
            _slurm.preflight(profile.target, (*argv, "--test-only"))
        except ObservationError as error:
            raise PlanError(str(error)) from error
        allocations.append(
            PlannedAllocation(
                allocation_id,
                tuple(task.key for task in group),
                len(group) * profile.resources.cpus_per_task,
                len(group) * profile.resources.memory_mib_per_task,
                len(group) * profile.resources.gpus_per_task,
                time_limit,
                script,
                argv,
            )
        )
    return replace(
        plan,
        allocations=tuple(allocations),
        digest=digest(
            {
                "decision": decision,
                "scripts": [allocation.script.decode("utf-8") for allocation in allocations],
            }
        ),
    )


def restore_plan(campaign: Campaign, document: object) -> SubmissionPlan:
    state = campaign._store.read()
    try:
        obj = object_fields(
            document,
            {
                "schema_version",
                "campaign_id",
                "revision",
                "profile",
                "selected_task_keys",
                "excluded_task_keys",
                "deferred_task_keys",
                "retry_task_keys",
                "duplicate_risk_task_keys",
                "tasks_per_allocation",
                "probe_required",
                "digest",
            },
        )
        if integer(obj["schema_version"]) != SCHEMA_VERSION:
            raise ValueError("unsupported plan schema")
        if (
            identifier(obj["campaign_id"], 32) != state.campaign_id
            or integer(obj["revision"]) != state.revision
        ):
            raise ValueError("plan is stale or foreign")
        selected, excluded, deferred, retry, ack = (
            keys(obj[name])
            for name in (
                "selected_task_keys",
                "excluded_task_keys",
                "deferred_task_keys",
                "retry_task_keys",
                "duplicate_risk_task_keys",
            )
        )
        roster = tuple(task.key for task in state.tasks)
        if set(selected) | set(excluded) | set(deferred) != set(roster) or len(
            selected + excluded + deferred
        ) != len(roster):
            raise ValueError("invalid plan roster partition")
        for selection in (selected, excluded, deferred):
            if tuple(key for key in roster if key in selection) != selection:
                raise ValueError("plan tasks are not in roster order")
        if (
            tuple(key for key in selected if key in retry) != retry
            or tuple(key for key in retry if key in ack) != ack
        ):
            raise ValueError("invalid retry references")
        plan = _build_plan(
            state,
            profile_from_document(obj["profile"]),
            selected,
            excluded,
            deferred,
            retry,
            ack,
            integer(obj["tasks_per_allocation"], 1),
            boolean(obj["probe_required"]),
        )
        if identifier(obj["digest"], 64) != plan.digest:
            raise ValueError("plan digest does not match regenerated scripts and decision")
        return plan
    except (ValueError, TypeError, ConfigurationError) as error:
        raise PlanError(f"invalid plan: {error}") from error


def sensitive_script_document(plan: SubmissionPlan) -> dict[str, object]:
    return {
        "allocations": [
            {
                "allocation_id": item.allocation_id,
                "script": item.script.decode("utf-8"),
                "argv": list(item.argv),
            }
            for item in plan.allocations
        ]
    }


def validation_document(results: tuple[ValidationResult, ...]) -> dict[str, object]:
    return {"time_specific": True, "results": [asdict(result) for result in results]}


def submit_document(result: SubmitResult) -> dict[str, object]:
    return {
        "receipts": [receipt_document(receipt) for receipt in result.receipts],
        "unresolved": [
            {
                "allocation_id": item.allocation_id,
                "task_keys": list(item.task_keys),
                "observed_receipt": None
                if item.observed_receipt is None
                else receipt_document(item.observed_receipt),
            }
            for item in result.unresolved
        ],
        "unattempted": [
            {"allocation_id": item.allocation_id, "task_keys": list(item.task_keys)}
            for item in result.unattempted
        ],
        "stop_reason": result.stop_reason,
    }


def _capacity(target: SlurmTarget, resources: ResourceRequest) -> int:
    if _duration_seconds(effective_time_limit(resources.time_limit), name="time_limit") > (
        _duration_seconds(effective_time_limit(target.max_time_limit), name="max_time_limit")
    ):
        raise PlanError("time_limit exceeds target ceiling")
    capacities = [
        target.max_tasks_per_allocation,
        target.max_cpus_per_allocation // resources.cpus_per_task,
        target.max_memory_mib_per_allocation // resources.memory_mib_per_task,
    ]
    if resources.gpus_per_task:
        if target.gpu_gres is None:
            raise PlanError("GPU work requires a target GPU GRES")
        capacities.append(target.max_gpus_per_allocation // resources.gpus_per_task)
    capacity = min(capacities)
    if capacity < 1:
        raise PlanError("one task exceeds target capacity")
    return capacity


def _balanced_groups(tasks: tuple[Task, ...], capacity: int) -> tuple[tuple[Task, ...], ...]:
    if not tasks:
        return ()
    count = len(tasks)
    group_count = (count + capacity - 1) // capacity
    small, larger = divmod(count, group_count)
    sizes = [small + 1] * larger + [small] * (group_count - larger)
    groups: list[tuple[Task, ...]] = []
    offset = 0
    for size in sizes:
        groups.append(tasks[offset : offset + size])
        offset += size
    return tuple(groups)
