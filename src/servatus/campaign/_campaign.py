"""The Campaign facade: the only module that composes the store, scheduler, probe, and clock.

Every operation reads durable state, gathers exactly the evidence it needs (result probe once,
scheduler evidence only for the Attempts that matter), decides with the pure policy, and commits
through the store. Scheduler contact never happens while the campaign lock is held.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable, Collection, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Self, cast

from ..errors import (
    Busy,
    ConfigurationError,
    Conflict,
    NotFound,
    PlanRefused,
    ReconciliationError,
    StalePlan,
    SubmissionInterrupted,
    Unavailable,
)
from ._config import Profile, StrPath, Target, Task, task_keys
from ._evidence import JobRef, Observation
from ._plan import Decision, Plan, PlannedAllocation, build_plan, capacity, decode_decision
from ._policy import Retry, decide, observation_scope, result_scope
from ._remote import Transport
from ._remote import connect as connect_target
from ._results import (
    LogSnapshot,
    Receipt,
    ShapeCheck,
    SubmitResult,
    UnattemptedAllocation,
    UnresolvedSubmission,
)
from ._scheduler import AttemptQuery, Scheduler
from ._script import log_path
from ._state import (
    AcceptanceState,
    Attempt,
    State,
    append,
    correct_outcome,
    create,
    record_intent,
    record_outcome,
    seal,
)
from ._status import ResultState, Status, classify_results, project
from ._store import Store

ResultProbe = Callable[[Sequence[Task]], Collection[str]]
"""Given Tasks, return the keys whose results are valid."""
Connect = Callable[[Target], Transport]
"""Given a Target, return the Transport that reaches its scheduler."""
Clock = Callable[[], datetime]

_ACCEPTED = AcceptanceState.ACCEPTED
_PRIVATE = object()


def _now() -> datetime:
    return datetime.now(UTC)


def ping(target: Target, *, connect: Connect | None = None) -> str:
    """Prove that ``target``'s scheduler answers; return its ``sbatch --version`` text.

    ``connect`` defaults to ``servatus.campaign.connect``. Failures raise ``Unavailable``.
    """
    if not isinstance(target, Target):
        raise ConfigurationError("ping needs a Target")
    return Scheduler((connect or connect_target)(target), target.slurm_bin).ping()


def _receipt(attempt: Attempt) -> Receipt:
    if attempt.job is None:
        raise Conflict(f"allocation {attempt.allocation_id} has no accepted job")
    return Receipt(attempt.allocation_id, attempt.job, attempt.task_keys)


def _merge(state: State, tasks: tuple[Task, ...]) -> State:
    known = {task.key: task for task in state.tasks}
    for task in tasks:
        if known.get(task.key, task) != task:
            raise Conflict(f"Task {task.key!r} differs from the registered Task with that key")
    unseen = [task for task in tasks if task.key not in known]
    return append(state, unseen) if unseen else state


def _interrupted(
    reason: str,
    error: Exception,
    receipts: list[Receipt],
    unresolved: tuple[UnresolvedSubmission, ...],
    unattempted: Sequence[PlannedAllocation],
) -> SubmissionInterrupted:
    result = SubmitResult(
        tuple(receipts),
        unresolved,
        tuple(UnattemptedAllocation(item.allocation_id, item.task_keys) for item in unattempted),
        reason,
    )
    return SubmissionInterrupted(f"submission interrupted: {reason} ({error})", result=result)


class Campaign:
    """A durable roster of Tasks and every Attempt to run them, in one owner-only directory.

    Use ``create``, ``open``, or ``ensure``. ``probe`` reports which Tasks have valid results;
    ``connect`` returns the ``Transport`` for a ``Target`` (default: SSH, or local without a
    host); ``clock`` returns the current aware time (default: UTC now).
    """

    __slots__ = ("_clock", "_connect", "_id", "_probe", "_store")

    def __init__(
        self,
        token: object,
        store: Store,
        probe: ResultProbe | None,
        connect: Connect | None,
        clock: Clock | None,
    ) -> None:
        if token is not _PRIVATE:
            raise TypeError("use Campaign.create, Campaign.open, or Campaign.ensure")
        for name, hook in (("probe", probe), ("connect", connect), ("clock", clock)):
            if hook is not None and not callable(hook):
                raise ConfigurationError(f"{name} must be callable")
        self._store = store
        self._probe = probe
        self._connect: Connect = connect or connect_target
        self._clock: Clock = clock or _now
        self._id = store.read().campaign_id

    # --- Construction -------------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        path: StrPath,
        tasks: Iterable[Task],
        *,
        appendable: bool = False,
        probe: ResultProbe | None = None,
        connect: Connect | None = None,
        clock: Clock | None = None,
    ) -> Self:
        """Create a new campaign (``Conflict`` if one exists). Sealed unless ``appendable``."""
        state = create(secrets.token_hex(16), tasks, appendable=appendable)
        return cls(_PRIVATE, Store.create(path, state), probe, connect, clock)

    @classmethod
    def open(
        cls,
        path: StrPath,
        *,
        probe: ResultProbe | None = None,
        connect: Connect | None = None,
        clock: Clock | None = None,
    ) -> Self:
        """Open an existing campaign (``NotFound`` if there is none)."""
        return cls(_PRIVATE, Store.open(path), probe, connect, clock)

    @classmethod
    def ensure(
        cls,
        path: StrPath,
        tasks: Iterable[Task],
        *,
        appendable: bool = True,
        probe: ResultProbe | None = None,
        connect: Connect | None = None,
        clock: Clock | None = None,
    ) -> Self:
        """Create the campaign (and missing parents), or open it and register unseen Tasks.

        Tasks already registered must be identical by key (``Conflict`` names the first
        mismatch); unseen Tasks are appended in order. ``appendable`` applies on creation.
        """
        candidate = create(secrets.token_hex(16), tasks, appendable=appendable)
        try:
            store = Store.open(path)
        except NotFound:
            try:
                return cls(
                    _PRIVATE, Store.create(path, candidate, parents=True), probe, connect, clock
                )
            except Conflict:  # created concurrently
                store = Store.open(path)
        store.update(lambda state: _merge(state, candidate.tasks))
        return cls(_PRIVATE, store, probe, connect, clock)

    def __repr__(self) -> str:
        return f"Campaign({str(self._store.path)!r})"

    # --- Roster -------------------------------------------------------------------------------

    @property
    def id(self) -> str:
        return self._id

    def tasks(self) -> tuple[Task, ...]:
        return self._store.read().tasks

    def append(self, tasks: Iterable[Task]) -> None:
        """Register new Tasks at the end of an appendable roster."""
        frozen = tuple(tasks)
        self._store.update(lambda state: append(state, frozen))

    def seal(self) -> None:
        """End authoring irreversibly. Sealing a sealed campaign changes nothing."""
        self._store.update(seal)

    # --- Observation --------------------------------------------------------------------------

    def status(self, *, scheduler: bool = True) -> Status:
        """Probe every Task and, unless ``scheduler`` is false, observe every accepted Attempt."""
        if type(scheduler) is not bool:
            raise ConfigurationError("scheduler must be a bool")
        state = self._store.read()
        results = self._results(state.tasks)
        everything = (item.allocation_id for item in state.attempts)
        observations = self._observe(state, everything) if scheduler else {}
        self._unchanged(state)
        return project(
            state, observations, results, scheduler_observed=scheduler, observed_at=self._time()
        )

    def read_log(
        self, *, task: str | None = None, allocation: str | None = None, max_bytes: int = 65_536
    ) -> LogSnapshot:
        """The last ``max_bytes`` (1 B to 1 MiB, checked before any contact) of an allocation
        log, or of one Task's step log.

        ``task`` alone reads that Task's step log in its current accepted allocation.
        """
        if task is None and allocation is None:
            raise ConfigurationError("read_log needs a task, an allocation, or both")
        state = self._store.read()
        if allocation is None:
            attempt = self._current(state, cast(str, task))
        else:
            attempt = state.attempt(allocation)
        job = _receipt(attempt).job
        slot = None
        if task is not None:
            if task not in attempt.task_keys:
                raise NotFound(f"Task {task!r} is not in allocation {attempt.allocation_id}")
            slot = attempt.task_keys.index(task)
        target = attempt.profile.target
        path = log_path(target.log_root, attempt.allocation_id, job.job_id, slot)
        content, truncated = self._scheduler(target).tail(path, max_bytes)
        return LogSnapshot(content, truncated, self._time())

    # --- Planning -----------------------------------------------------------------------------

    def plan(
        self,
        profile: Profile,
        *,
        retry: Collection[str] | Retry = (),
        allow_duplicate_risk: Collection[str] = (),
        only: Collection[str] | None = None,
        tasks_per_allocation: int | None = None,
    ) -> Plan:
        """Decide which Tasks to submit with ``profile`` and pack them into allocations.

        Only Tasks never accepted, or requested for retry, are probed; scheduler evidence is
        gathered only for the accepted Attempts of retry candidates.
        """
        if not isinstance(profile, Profile):
            raise ConfigurationError("profile must be a Profile")
        chosen = retry if isinstance(retry, Retry) else task_keys(retry, "retry")
        acknowledged = task_keys(allow_duplicate_risk, "allow_duplicate_risk")
        scope = None if only is None else task_keys(only, "only")
        if chosen is Retry.INCOMPLETE and self._probe is None:
            raise ConfigurationError("Retry.INCOMPLETE needs a result probe")
        cap = capacity(profile.target, profile.resources, tasks_per_allocation)
        state = self._store.read()
        results = self._results(result_scope(state, chosen, scope))
        observations = self._observe(state, observation_scope(state, chosen, scope))
        self._unchanged(state)
        selection = decide(
            state, observations, results, retry=chosen, duplicate_risk=acknowledged, only=scope
        )
        limit = profile.target.max_allocations_per_submit
        bound = len(selection.selected) if limit is None else cap * limit
        decision = Decision(
            campaign_id=state.campaign_id,
            revision=state.revision,
            nonce=secrets.token_hex(16),
            profile=profile,
            selected=selection.selected[:bound],
            held=selection.held,
            deferred=selection.selected[bound:],
            retry=selection.retry,
            duplicate_risk=selection.duplicate_risk,
            tasks_per_allocation=cap,
            probe_required=self._probe is not None,
        )
        return build_plan(state, decision)

    def load_plan(self, data: bytes) -> Plan:
        """Rebuild a saved plan (``Plan.to_json``) against the current campaign."""
        decision, digest = decode_decision(data)
        plan = build_plan(self._store.read(), decision)
        if plan.digest != digest:
            raise ConfigurationError("the plan digest does not match its decision")
        return plan

    def validate(self, plan: Plan) -> tuple[ShapeCheck, ...]:
        """Ask ``sbatch --test-only`` once per distinct allocation shape. Records nothing."""
        plan = self._verified(plan)
        if not plan.allocations:
            return ()
        scheduler = self._scheduler(plan.decision.profile.target)
        shapes: dict[int, PlannedAllocation] = {}
        for item in plan.allocations:
            shapes.setdefault(len(item.task_keys), item)
        return tuple(
            ShapeCheck(
                count,
                item.cpus,
                item.memory_mib,
                item.gpus,
                item.time_limit,
                *scheduler.test_only(item.argv, item.script),
            )
            for count, item in shapes.items()
        )

    # --- Submission ---------------------------------------------------------------------------

    def submit(self, plan: Plan) -> SubmitResult:
        """Submit exactly the reviewed plan, recording durable intent before each ``sbatch``.

        The plan is rebuilt and compared first, the scheduler is pinged, and every planned Task
        must still be eligible; all of that happens before any intent. A stop after that raises
        ``SubmissionInterrupted`` whose ``result`` lists receipts, unresolved allocations, and
        unattempted allocations.
        """
        plan = self._verified(plan)
        decision = plan.decision
        if not plan.allocations:
            return SubmitResult(())
        if decision.probe_required and self._probe is None:
            raise ConfigurationError(
                "this plan was made with a result probe; submit it from a Campaign that has one"
            )
        scheduler = self._scheduler(decision.profile.target)
        scheduler.ping()
        self._still_eligible(decision)
        receipts: list[Receipt] = []
        expected: int | None = decision.revision
        for index, item in enumerate(plan.allocations):
            later = plan.allocations[index + 1 :]
            unresolved = (UnresolvedSubmission(item.allocation_id, item.task_keys),)
            try:
                claimed = self._claim(decision, item, expected)
            except Exception as error:
                if self._recorded(item.allocation_id):  # durable, though the commit failed
                    stop = "recording the intent failed after it became durable"
                    raise _interrupted(stop, error, receipts, unresolved, later) from error
                stop = "the next intent was not recorded"
                raise _interrupted(stop, error, receipts, (), plan.allocations[index:]) from error
            try:
                job = scheduler.submit(item.argv, item.script)
            except Exception as error:
                stop = "scheduler acceptance is unresolved"
                raise _interrupted(stop, error, receipts, unresolved, later) from error
            try:
                state = self._resolve(item.allocation_id, job)
            except Exception as error:
                if (receipt := self._durable_receipt(item.allocation_id, job)) is not None:
                    receipts.append(receipt)
                    stop = "recording the receipt failed after it became durable"
                    raise _interrupted(stop, error, receipts, (), later) from error
                stop = f"Slurm accepted job {job} but recording the receipt failed"
                observed = UnresolvedSubmission(item.allocation_id, item.task_keys, job)
                raise _interrupted(stop, error, receipts, (observed,), later) from error
            attempt = state.attempt(item.allocation_id)
            receipts.append(_receipt(attempt))
            # Continue only if this receipt is the one change since the intent; any other
            # change makes the next claim stale.
            unchanged = attempt.outcome_revision == state.revision == claimed + 1
            expected = state.revision if unchanged else None
        return SubmitResult(tuple(receipts))

    def reconcile(self, allocation_id: str) -> Receipt:
        """Record the one job Slurm holds for an unresolved allocation, from its original target."""
        attempt = self._store.read().attempt(allocation_id)
        if attempt.acceptance is _ACCEPTED:
            return _receipt(attempt)
        if attempt.acceptance is AcceptanceState.NOT_SUBMITTED:
            raise Conflict(f"allocation {allocation_id} is recorded as not submitted")
        target = attempt.profile.target
        job = self._scheduler(target).identify(allocation_id, attempt.intent_at)
        return _receipt(self._resolve(allocation_id, job).attempt(allocation_id))

    def mark_accepted(
        self, allocation_id: str, job_id: int, *, cluster: str | None = None
    ) -> Receipt:
        """Record a job the operator found for an unresolved allocation.

        An allocation already recorded as not submitted is corrected only when its original
        target proves that exactly this job carries the allocation's identity.
        """
        job = JobRef(job_id, cluster)
        attempt = self._store.read().attempt(allocation_id)
        if attempt.acceptance is not AcceptanceState.NOT_SUBMITTED:
            return _receipt(self._resolve(allocation_id, job).attempt(allocation_id))
        scheduler = self._scheduler(attempt.profile.target)
        try:
            found = scheduler.identify(allocation_id, attempt.intent_at)
        except ReconciliationError as error:
            found, detail = None, str(error)
        else:
            detail = f"Slurm holds job {found}"
        if found != job:
            raise Conflict(
                f"allocation {allocation_id} is recorded as not submitted and Slurm does not "
                f"prove job {job} for it ({detail})"
            )
        state = self._store.update(lambda current: correct_outcome(current, allocation_id, job))
        return _receipt(state.attempt(allocation_id))

    def mark_not_submitted(self, allocation_id: str) -> None:
        """Record that an unresolved allocation never reached Slurm, once Slurm proves it.

        The original target is asked first. ``Conflict`` when any job carries the allocation's
        identity (record it with ``reconcile`` or ``mark_accepted`` instead); a failed query
        raises ``Unavailable`` and records nothing.
        """
        attempt = self._store.read().attempt(allocation_id)
        if attempt.acceptance is AcceptanceState.UNRESOLVED:
            scheduler = self._scheduler(attempt.profile.target)
            if found := scheduler.find(allocation_id, attempt.intent_at):
                raise Conflict(
                    f"Slurm holds job {', '.join(map(str, found))} for allocation "
                    f"{allocation_id}; record it with reconcile or mark-accepted instead"
                )
        self._store.update(lambda state: record_outcome(state, allocation_id, None))

    def cancel(
        self, *, tasks: Collection[str] | None = None, allocations: Collection[str] | None = None
    ) -> tuple[Receipt, ...]:
        """``scancel`` accepted allocations that are not known to be terminal.

        ``tasks`` selects every accepted allocation of each Task. Returns the receipts of the
        allocations that were asked to stop; packed allocations stop all their Tasks.
        """
        keys = () if tasks is None else task_keys(tasks, "tasks")
        identities = () if allocations is None else task_keys(allocations, "allocations")
        if not keys and not identities:
            raise ConfigurationError("cancel needs Task keys or allocation ids")
        state = self._store.read()
        chosen = {name: _receipt(state.attempt(name)) for name in identities}
        roster = {task.key for task in state.tasks}
        for key in keys:
            if key not in roster:
                raise NotFound(f"unknown Task: {key!r}")
            owned = [
                item
                for item in state.attempts
                if item.acceptance is _ACCEPTED and key in item.task_keys
            ]
            if not owned:
                raise Conflict(f"Task {key!r} has no accepted allocation to cancel")
            chosen.update((item.allocation_id, _receipt(item)) for item in owned)
        observations = self._observe(state, tuple(chosen))
        cancelled: list[Receipt] = []
        failed: list[str] = []
        for attempt in state.attempts:
            receipt = chosen.get(attempt.allocation_id)
            evidence = observations[attempt.allocation_id].allocation if receipt else None
            if receipt is None or (evidence and evidence.state.terminal and not evidence.retained):
                continue
            scheduler = self._scheduler(attempt.profile.target)
            try:
                scheduler.cancel(attempt.allocation_id, receipt.job)
            except Unavailable as error:
                failed.append(f"{attempt.allocation_id} ({error})")
            else:
                cancelled.append(receipt)
        if failed:
            done = ", ".join(item.allocation_id for item in cancelled) or "none"
            raise Unavailable(f"cancel failed for {'; '.join(failed)}; cancelled: {done}")
        return tuple(cancelled)

    # --- Internals ----------------------------------------------------------------------------

    def _time(self) -> datetime:
        moment = self._clock()
        if not isinstance(moment, datetime) or moment.utcoffset() is None:
            raise ConfigurationError("clock must return an aware datetime")
        return moment.astimezone(UTC)

    def _scheduler(self, target: Target) -> Scheduler:
        return Scheduler(self._connect(target), target.slurm_bin)

    def _results(self, tasks: tuple[Task, ...]) -> dict[str, ResultState]:
        """Call the probe once for ``tasks``; without a probe every result is UNOBSERVED."""
        if self._probe is None or not tasks:
            return {}
        return classify_results((task.key for task in tasks), self._probe(tasks))

    def _observe(self, state: State, allocation_ids: Iterable[str]) -> dict[str, Observation]:
        """Observe the accepted Attempts among ``allocation_ids``, one scheduler per original
        (host, slurm_bin) route."""
        wanted = set(allocation_ids)
        routes: dict[tuple[str | None, PurePosixPath], tuple[Target, list[AttemptQuery]]] = {}
        for attempt in state.attempts:
            if attempt.allocation_id in wanted and attempt.job is not None:
                target = attempt.profile.target
                route = routes.setdefault((target.host, target.slurm_bin), (target, []))
                query = AttemptQuery(
                    attempt.allocation_id, attempt.job, attempt.intent_at, len(attempt.task_keys)
                )
                route[1].append(query)
        observed: dict[str, Observation] = {}
        for target, queries in routes.values():
            observed.update(self._scheduler(target).observe(queries))
        return observed

    def _unchanged(self, state: State) -> None:
        if self._store.read().revision != state.revision:
            raise Busy("the campaign changed while it was being observed; try again")

    def _current(self, state: State, key: str) -> Attempt:
        if key not in {task.key for task in state.tasks}:
            raise NotFound(f"unknown Task: {key!r}")
        for attempt in reversed(state.attempts):
            if attempt.acceptance is _ACCEPTED and key in attempt.task_keys:
                return attempt
        raise NotFound(f"Task {key!r} has no accepted allocation")

    def _verified(self, plan: Plan) -> Plan:
        """Rebuild ``plan`` from its Decision; refuse it unless it is exactly what was reviewed."""
        if not isinstance(plan, Plan) or not isinstance(plan.decision, Decision):
            raise ConfigurationError("plan must be a Plan")
        rebuilt = build_plan(self._store.read(), plan.decision)
        if rebuilt != plan:
            raise ConfigurationError("the plan does not match its decision; plan again")
        return rebuilt

    def _still_eligible(self, decision: Decision) -> None:
        """Probe and observe once for the planned Tasks; each must still be selected."""
        state = self._store.read()
        if state.revision != decision.revision:
            raise StalePlan("the campaign changed after planning; plan again")
        keys = decision.selected
        retry = tuple(key for key in decision.retry if key in keys)
        risk = tuple(key for key in decision.duplicate_risk if key in keys)
        results = self._results(tuple(task for task in state.tasks if task.key in keys))
        observations = self._observe(state, observation_scope(state, retry, keys))
        try:
            selection = decide(
                state, observations, results, retry=retry, duplicate_risk=risk, only=keys
            )
        except PlanRefused as error:
            raise StalePlan(f"the plan no longer holds ({error}); plan again") from error
        if selection.selected != keys:
            held = ", ".join(
                f"{key!r} ({selection.held[key]})" for key in keys if key in selection.held
            )
            raise StalePlan(f"planned Tasks are no longer eligible: {held}; plan again")

    def _claim(self, decision: Decision, item: PlannedAllocation, expected: int | None) -> int:
        """Record intent if the campaign is still at ``expected``; return the committed revision.

        ``expected`` is None once the campaign is known to have changed during submission.
        """
        keys = item.task_keys

        def claim(state: State) -> State:
            if expected is None or state.revision != expected:
                raise StalePlan("the campaign changed during submission; plan again")
            changed, _ = record_intent(
                state,
                allocation_id=item.allocation_id,
                task_keys=keys,
                profile=decision.profile,
                retry=tuple(key for key in keys if key in decision.retry),
                duplicate_risk=tuple(key for key in keys if key in decision.duplicate_risk),
                at=self._time(),
            )
            return changed

        return self._store.update(claim).revision

    def _recorded(self, allocation_id: str) -> bool:
        """Whether an intent is durable; an unreadable campaign counts as recorded."""
        try:
            return any(item.allocation_id == allocation_id for item in self._store.read().attempts)
        except Exception:
            return True

    def _durable_receipt(self, allocation_id: str, job: JobRef) -> Receipt | None:
        """The receipt, if acceptance of ``job`` is durable though its commit reported failure."""
        try:
            attempt = self._store.read().attempt(allocation_id)
        except Exception:
            return None
        return _receipt(attempt) if attempt.job == job else None

    def _resolve(self, allocation_id: str, job: JobRef) -> State:
        """Record acceptance of ``job``; return the committed state."""
        return self._store.update(lambda state: record_outcome(state, allocation_id, job))
