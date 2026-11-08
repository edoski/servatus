"""The Campaign facade: the only module that composes the store, scheduler, probe, and clock.

Every operation reads durable state, gathers exactly the evidence it needs (result probe once,
scheduler evidence only for the Attempts that matter), decides with the pure policy, and commits
through the store. Scheduler contact never happens while the campaign lock is held.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable, Collection, Iterable, Sequence
from datetime import UTC, datetime, timedelta
from functools import reduce
from pathlib import PurePosixPath
from typing import Self, cast

from .._fs import StrPath
from ..errors import (
    Busy,
    ConfigurationError,
    Conflict,
    NotFound,
    PlanRefused,
    StalePlan,
    SubmissionInterrupted,
    Unavailable,
)
from ._config import Profile, Target, Task, task_keys
from ._evidence import Marker, Observation, Observed, QueueRow, evidenced_job, observe, queue_row
from ._plan import Decision, Plan, PlannedAllocation, batch, build_plan, capacity, decode_decision
from ._policy import Request, Retry, decide, scope
from ._remote import Transport
from ._remote import connect as connect_target
from ._results import (
    LogSnapshot,
    Receipt,
    ReconcileResult,
    ShapeCheck,
    SubmitResult,
    UnattemptedAllocation,
    UnresolvedSubmission,
    WaitingAllocation,
)
from ._scheduler import Scheduler
from ._script import log_path
from ._state import (
    AcceptanceState,
    Attempt,
    Link,
    State,
    append,
    create,
    history,
    record_intent,
    record_outcome,
    seal,
)
from ._status import ResultState, Status, classify_results, current, project
from ._store import Store

ResultProbe = Callable[[Sequence[Task]], Collection[str]]
"""Given Tasks, return the keys whose results are valid."""
Connect = Callable[[Target], Transport]
"""Given a Target, return the Transport that reaches its scheduler."""
Clock = Callable[[], datetime]

_ACCEPTED = AcceptanceState.ACCEPTED
_PRIVATE = object()
NOT_SUBMITTED_AFTER = timedelta(minutes=15)
"""How long an unresolved allocation without any evidence waits before ``reconcile`` records it
as not submitted: long past any ``sbatch`` still in flight when its reply was lost."""


def _now() -> datetime:
    return datetime.now(UTC)


def _receipt(attempt: Attempt) -> Receipt:
    if attempt.job is None:
        raise Conflict(f"allocation {attempt.allocation_id} has no accepted job")
    return Receipt(attempt.allocation_id, attempt.job, attempt.task_keys)


def _outcomes(state: State, outcomes: Sequence[tuple[str, int | None]]) -> State:
    return reduce(lambda current, item: record_outcome(current, *item), outcomes, state)


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
        observed = self._observe(state, everything) if scheduler else Observed({}, {})
        self._unchanged(state)
        return project(
            state, observed, results, scheduler_observed=scheduler, observed_at=self._time()
        )

    def read_log(
        self, *, task: str | None = None, allocation: str | None = None, max_bytes: int = 65_536
    ) -> LogSnapshot:
        """The last ``max_bytes`` (1 B to 1 MiB, checked before any contact) of an allocation
        log, or of one Task's step log.

        ``task`` alone reads that Task's step log in its current allocation; in a chain, finding
        it observes the chain's later links.
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
        path = log_path(target.log_root, attempt.allocation_id, job, slot)
        content, truncated = self._scheduler(target).tail(path, max_bytes)
        return LogSnapshot(content, truncated, self._time())

    # --- Planning -----------------------------------------------------------------------------

    def plan(
        self,
        profile: Profile,
        *,
        retry: Collection[str] | Retry = (),
        only: Collection[str] | None = None,
        tasks_per_allocation: int | None = None,
    ) -> Plan:
        """Decide which Tasks to submit with ``profile`` and pack them into allocations.

        Only Tasks never accepted, or requested for retry, are probed; scheduler evidence is
        gathered only for the accepted Attempts of retry candidates.
        """
        if not isinstance(profile, Profile):
            raise ConfigurationError("profile must be a Profile")
        request = Request.parse(retry, only)
        if request.retry is Retry.INCOMPLETE and self._probe is None:
            raise ConfigurationError("Retry.INCOMPLETE needs a result probe")
        cap = capacity(profile.target, profile.resources, tasks_per_allocation)
        state = self._store.read()
        observed, results = self._consult(state, request)
        self._unchanged(state)
        selection = decide(state, observed, results, request)
        selected, deferred = batch(selection.selected, profile.target, cap)
        decision = Decision(
            campaign_id=state.campaign_id,
            revision=state.revision,
            nonce=secrets.token_hex(16),
            profile=profile,
            selected=selected,
            held=selection.held,
            deferred=deferred,
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
        for index, item in enumerate(plan.allocations):
            later = plan.allocations[index + 1 :]
            unresolved = (UnresolvedSubmission(item.allocation_id, item.task_keys),)
            try:
                # Each allocation adds an intent and a receipt; any other change is stale.
                self._claim(decision, item, decision.revision + 2 * index)
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
            receipts.append(_receipt(state.attempt(item.allocation_id)))
        return SubmitResult(tuple(receipts))

    def reconcile(self) -> ReconcileResult:
        """Resolve every unresolved allocation from Slurm's evidence, idempotently.

        A queue row, else a run marker, records acceptance of its job. Without either, an
        allocation whose intent is at least ``NOT_SUBMITTED_AFTER`` old is recorded as not
        submitted; a younger one stays unresolved and is reported as ``waiting``.
        """
        now = self._time()
        state = self._store.read()
        unresolved = [
            item for item in state.attempts if item.acceptance is AcceptanceState.UNRESOLVED
        ]
        rows, markers = self._evidence(unresolved, unresolved)
        outcomes: list[tuple[str, int | None]] = []
        waiting: list[WaitingAllocation] = []
        for attempt in unresolved:
            name = attempt.allocation_id
            job = evidenced_job(name, rows[name], markers[name])
            decidable_at = attempt.intent_at + NOT_SUBMITTED_AFTER
            if job is not None or now >= decidable_at:
                outcomes.append((name, job))
            else:
                waiting.append(WaitingAllocation(name, attempt.task_keys, decidable_at))
        if outcomes:
            state = self._store.update(lambda current: _outcomes(current, outcomes))
        resolved = [state.attempt(name) for name, _ in outcomes]
        return ReconcileResult(
            tuple(_receipt(item) for item in resolved if item.job is not None),
            tuple(
                UnattemptedAllocation(item.allocation_id, item.task_keys)
                for item in resolved
                if item.job is None
            ),
            tuple(waiting),
        )

    def cancel(
        self, *, tasks: Collection[str] | None = None, allocations: Collection[str] | None = None
    ) -> tuple[Receipt, ...]:
        """``scancel`` the chosen allocations that Slurm still lists as queued or running.

        ``tasks`` selects every allocation of each Task that was accepted or recorded as not
        submitted. Returns a receipt (with the job Slurm listed) for each allocation that was
        asked to stop; packed allocations stop all their Tasks.
        """
        keys = () if tasks is None else task_keys(tasks, "tasks")
        identities = () if allocations is None else task_keys(allocations, "allocations")
        if not keys and not identities:
            raise ConfigurationError("cancel needs Task keys or allocation ids")
        state = self._store.read()
        roster = {task.key for task in state.tasks}
        for key in keys:
            if key not in roster:
                raise NotFound(f"unknown Task: {key!r}")
        for name in identities:
            if state.attempt(name).acceptance is AcceptanceState.UNRESOLVED:
                raise Conflict(f"allocation {name} is unresolved; reconcile it first")
        chosen = [
            item
            for item in state.attempts
            if item.acceptance is not AcceptanceState.UNRESOLVED
            and (item.allocation_id in identities or set(keys).intersection(item.task_keys))
        ]
        if missing := [key for key in keys if not any(key in item.task_keys for item in chosen)]:
            raise Conflict(f"Task {missing[0]!r} has no submitted allocation to cancel")
        rows, _ = self._evidence(chosen)
        cancelled: list[Receipt] = []
        failed: list[str] = []
        for attempt in chosen:
            row = queue_row(attempt.allocation_id, rows[attempt.allocation_id], attempt.job)
            if row is None or row.settling:
                continue
            try:
                self._scheduler(attempt.profile.target).cancel(attempt.allocation_id)
            except Unavailable as error:
                failed.append(f"{attempt.allocation_id} ({error})")
            else:
                cancelled.append(Receipt(attempt.allocation_id, row.job, attempt.task_keys))
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
        return Scheduler(self._connect(target), target)

    def _results(self, tasks: tuple[Task, ...]) -> dict[str, ResultState]:
        """Call the probe once for ``tasks``; without a probe every result is UNOBSERVED."""
        if self._probe is None or not tasks:
            return {}
        return classify_results((task.key for task in tasks), self._probe(tasks))

    def _evidence(
        self, queued: Sequence[Attempt], marked: Sequence[Attempt] = ()
    ) -> tuple[dict[str, tuple[QueueRow, ...]], dict[str, tuple[Marker, ...]]]:
        """Queue rows of ``queued`` (query batches per original host and ``slurm_bin``) and run
        markers of ``marked`` (one directory read per original host and ``log_root``)."""
        queues: dict[tuple[str | None, PurePosixPath], tuple[Target, list[str]]] = {}
        roots: dict[tuple[str | None, PurePosixPath], tuple[Target, dict[str, int]]] = {}
        for attempt in queued:
            target = attempt.profile.target
            queue = queues.setdefault((target.host, target.slurm_bin), (target, []))
            queue[1].append(attempt.allocation_id)
        for attempt in marked:
            target = attempt.profile.target
            root = roots.setdefault((target.host, target.log_root), (target, {}))
            root[1][attempt.allocation_id] = len(attempt.task_keys)
        rows: dict[str, tuple[QueueRow, ...]] = {}
        for target, names in queues.values():
            rows.update(self._scheduler(target).queue(names))
        found: dict[str, tuple[Marker, ...]] = {}
        for target, counts in roots.values():
            found.update(self._scheduler(target).markers(counts))
        return rows, found

    def _observe(self, state: State, allocation_ids: Iterable[str]) -> Observed:
        """Observe the accepted and not-submitted Attempts among ``allocation_ids``: accepted
        ones from their queue rows and run markers, not-submitted ones from their queue rows."""
        wanted = set(allocation_ids)
        chosen = [
            item
            for item in state.attempts
            if item.allocation_id in wanted and item.acceptance is not AcceptanceState.UNRESOLVED
        ]
        accepted = [item for item in chosen if item.acceptance is _ACCEPTED]
        rows, markers = self._evidence(chosen, accepted)
        observations: dict[str, Observation] = {}
        strays: dict[str, int] = {}
        for attempt in chosen:
            name = attempt.allocation_id
            if attempt.job is not None:
                observations[name] = observe(
                    name,
                    attempt.job,
                    len(attempt.task_keys),
                    rows[name],
                    markers[name],
                    gated=attempt.link.index > 1,
                )
            elif (row := queue_row(name, rows[name], None)) is not None:
                strays[name] = row.job
        return Observed(observations, strays)

    def _unchanged(self, state: State) -> None:
        if self._store.read().revision != state.revision:
            raise Busy("the campaign changed while it was being observed; try again")

    def _current(self, state: State, key: str) -> Attempt:
        if (past := history(state).get(key)) is None:
            raise NotFound(f"unknown Task: {key!r}")
        chained = [attempt.allocation_id for attempt, _ in past.accepted if attempt.link.index > 1]
        observed = self._observe(state, chained) if chained else Observed({}, {})
        found = current(past, observed)
        if found is None:
            raise NotFound(f"Task {key!r} has no accepted allocation")
        return found[0]

    def _verified(self, plan: Plan) -> Plan:
        """Rebuild ``plan`` the way a saved plan is loaded; refuse it unless it is exactly what
        was reviewed."""
        if not isinstance(plan, Plan) or not isinstance(plan.decision, Decision):
            raise ConfigurationError("plan must be a Plan")
        try:
            data = plan.to_json()
        except ValueError as error:
            raise ConfigurationError(f"invalid plan: {error}") from None
        rebuilt = self.load_plan(data)
        if rebuilt != plan:
            raise ConfigurationError("the plan does not match its decision; plan again")
        return rebuilt

    def _still_eligible(self, decision: Decision) -> None:
        """Probe and observe once for the planned Tasks; each must still be selected."""
        state = self._store.read()
        if state.revision != decision.revision:
            raise StalePlan("the campaign changed after planning; plan again")
        keys = decision.selected
        past = history(state)
        request = Request(frozenset(key for key in keys if past[key].accepted), frozenset(keys))
        observed, results = self._consult(state, request)
        try:
            selection = decide(state, observed, results, request)
        except PlanRefused as error:
            raise StalePlan(f"the plan no longer holds ({error}); plan again") from error
        if selection.selected != keys:
            held = ", ".join(
                f"{key!r} ({selection.held[key]})" for key in keys if key in selection.held
            )
            raise StalePlan(f"planned Tasks are no longer eligible: {held}; plan again")

    def _consult(self, state: State, request: Request) -> tuple[Observed, dict[str, ResultState]]:
        """Probe and observe, once, exactly what ``decide`` may consult for ``request``."""
        tasks, allocation_ids = scope(state, request)
        results = self._results(tasks)
        return self._observe(state, allocation_ids), results

    def _claim(self, decision: Decision, item: PlannedAllocation, expected: int) -> None:
        """Record intent if the campaign is still at revision ``expected``."""
        keys = item.task_keys

        def claim(state: State) -> State:
            if state.revision != expected:
                raise StalePlan("the campaign changed during submission; plan again")
            return record_intent(
                state,
                allocation_id=item.allocation_id,
                link=Link(item.allocation_id, 1, 1),
                task_keys=keys,
                profile=decision.profile,
                at=self._time(),
            )

        self._store.update(claim)

    def _recorded(self, allocation_id: str) -> bool:
        """Whether an intent is durable; an unreadable campaign counts as recorded."""
        try:
            return any(item.allocation_id == allocation_id for item in self._store.read().attempts)
        except Exception:
            return True

    def _durable_receipt(self, allocation_id: str, job: int) -> Receipt | None:
        """The receipt, if acceptance of ``job`` is durable though its commit reported failure."""
        try:
            attempt = self._store.read().attempt(allocation_id)
        except Exception:
            return None
        return _receipt(attempt) if attempt.job == job else None

    def _resolve(self, allocation_id: str, job: int) -> State:
        """Record acceptance of ``job``; return the committed state."""
        return self._store.update(lambda state: record_outcome(state, allocation_id, job))
