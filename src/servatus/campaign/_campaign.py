"""The Campaign facade: the only module that composes the store, scheduler, probe, and clock.

Every operation reads durable state, gathers exactly the evidence it needs (result probe once,
scheduler evidence only for the Attempts that matter), decides with the pure policy, and commits
through the store. Scheduler contact never happens while the campaign lock is held.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from functools import reduce
from pathlib import PurePosixPath
from typing import Self, cast

from .._fs import StrPath
from ..errors import (
    Busy,
    ConfigurationError,
    Conflict,
    EvidenceConflict,
    NotFound,
    PlanRefused,
    StalePlan,
    SubmissionInterrupted,
    Unavailable,
)
from ._config import Continuation, Profile, Target, Task, task_keys
from ._evidence import (
    Marker,
    Observation,
    Observed,
    QueueRow,
    Void,
    evidenced_job,
    observe,
    observe_step,
    queue_row,
)
from ._plan import (
    Decision,
    Plan,
    PlannedAllocation,
    batch,
    build_plan,
    capacity,
    decode_decision,
    link_profile,
)
from ._policy import Hold, Request, Retry, decide, scope
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
from ._script import dependency_options, log_path
from ._state import (
    AcceptanceState,
    Attempt,
    Kind,
    State,
    append,
    chain_jobs,
    create,
    history,
    record_intent,
    record_outcome,
    seal,
)
from ._status import ResultState, Status, classify_results, current, project, run_time
from ._store import Store

ResultProbe = Callable[[Sequence[Task]], Collection[str]]
"""Given Tasks, return the keys whose results are valid."""
Connect = Callable[[Target], Transport]
"""Given a Target, return the Transport that reaches its scheduler."""
Clock = Callable[[], datetime]

_ACCEPTED = AcceptanceState.ACCEPTED
_PRIVATE = object()
_ZERO = timedelta(0)
_PLAIN = Continuation()
_LATER_LINK = ("--kill-on-invalid-dep=yes",)
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
    """``state`` with each outcome recorded for the Attempts still unresolved: a concurrent
    submission may have resolved one since."""
    unresolved = AcceptanceState.UNRESOLVED
    pending = [item for item in outcomes if state.attempt(item[0]).acceptance is unresolved]
    return reduce(lambda current, item: record_outcome(current, *item), pending, state)


def _merge(state: State, tasks: tuple[Task, ...]) -> State:
    known = {task.key: task for task in state.tasks}
    for task in tasks:
        if known.get(task.key, task) != task:
            raise Conflict(f"Task {task.key!r} differs from the registered Task with that key")
    unseen = [task for task in tasks if task.key not in known]
    return append(state, unseen) if unseen else state


def _budgeted(
    request: Request, continuation: Continuation, profile: Profile, spent: Mapping[str, timedelta]
) -> Request:
    """``request`` under ``continuation``'s budget: the Tasks with no room for one more link."""
    if continuation.budget is None:
        return request
    exhausted = frozenset(
        key for key, ran in spent.items() if not continuation.link_limits(profile.resources, ran)
    )
    return replace(request, exhausted=exhausted)


def _unique(attempts: Iterable[Attempt]) -> list[Attempt]:
    return list({attempt.allocation_id: attempt for attempt in attempts}.values())


def _withins(steps: Iterable[Attempt]) -> list[str]:
    return list(dict.fromkeys(cast(str, item.within) for item in steps))


def _size(item: PlannedAllocation) -> int:
    """Tasks per allocation, or a lease's slots."""
    return len(item.task_keys) if item.slots is None else item.slots


def _serving(observation: Observation) -> bool:
    """Whether a lease takes new orders: Slurm lists it, it has not ended, and its pre-limit
    signal has not arrived."""
    return not observation.ended and observation.limit_at is None


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
        slot = 0 if attempt.kind is Kind.STEP else None
        if task is not None:
            if task not in attempt.task_keys:
                raise NotFound(f"Task {task!r} is not in allocation {attempt.allocation_id}")
            slot = attempt.task_keys.index(task)
        target = attempt.profile.target
        path = log_path(target.log_root, attempt.allocation_id, job, slot)
        content, truncated = self._scheduler(target).tail(path, max_bytes)
        return LogSnapshot(content, truncated, self._time())

    # --- Planning -----------------------------------------------------------------------------

    def hold(
        self, profile: Profile, *, slots: int | None = None, continuation: Continuation = _PLAIN
    ) -> Plan:
        """A plan of one lease: ``slots`` (default: as many as fit) copies of the profile's
        per-Task Resources, held for its whole time limit, selecting no Task. ``continuation``
        may set ``max_links`` to queue that many leases, each starting when the one before it
        ended; a budget is refused. Needs ``signal_before_end``. Observes nothing."""
        if not isinstance(profile, Profile):
            raise ConfigurationError("profile must be a Profile")
        if not isinstance(continuation, Continuation):
            raise ConfigurationError("continuation must be a Continuation")
        if continuation.budget is not None:
            raise ConfigurationError("a lease takes no budget; its steps may")
        if profile.resources.signal_before_end is None:
            raise ConfigurationError("a lease needs resources.signal_before_end")
        size = capacity(profile.target, profile.resources, slots)
        continuation.check(profile.resources)
        state = self._store.read()
        decision = Decision(
            campaign_id=state.campaign_id,
            revision=state.revision,
            nonce=secrets.token_hex(16),
            profile=profile,
            selected=(),
            held={task.key: Hold.NOT_REQUESTED for task in state.tasks},
            deferred=(),
            tasks_per_allocation=1,
            probe_required=False,
            continuation=continuation,
            spent={},
            extensions={},
            slots=size,
        )
        return build_plan(state, decision)

    def plan(
        self,
        profile: Profile | None = None,
        *,
        lease: str | None = None,
        slots: int | None = None,
        retry: Collection[str] | Retry = (),
        only: Collection[str] | None = None,
        tasks_per_allocation: int | None = None,
        continuation: Continuation = _PLAIN,
    ) -> Plan:
        """Decide which Tasks to submit with ``profile`` and pack them into chains of
        allocations, one link each unless ``continuation`` asks for more.

        Only Tasks never accepted, or requested for retry, are probed; scheduler evidence is
        gathered only for the accepted Attempts of retry candidates. An active continuation
        measures each candidate's run time from that evidence and records it in the Decision.

        With ``lease`` instead of a profile (the id of any link of a lease of this campaign),
        each selected Task starts as a step using ``slots`` (default 1) of the lease's current
        link: the first one Slurm lists that has not ended or reached its pre-limit signal
        (``Conflict`` when there is none). Steps use the lease's Profile, always learn their run
        time, and are never deferred; ``continuation`` may carry a budget but not ``max_links``.
        """
        if not isinstance(continuation, Continuation):
            raise ConfigurationError("continuation must be a Continuation")
        if lease is None:
            if not isinstance(profile, Profile):
                raise ConfigurationError("profile must be a Profile")
            if slots is not None:
                raise ConfigurationError("slots needs a lease")
        elif profile is not None or tasks_per_allocation is not None:
            raise ConfigurationError("a step plan takes its Profile from its lease, one Task each")
        elif continuation.max_links is not None:
            raise ConfigurationError("steps do not continue across links; plan without max_links")
        elif slots is not None and (type(slots) is not int or slots < 1):
            raise ConfigurationError("slots must be a positive integer")
        request = Request.parse(retry, only)
        if request.retry is Retry.INCOMPLETE and self._probe is None:
            raise ConfigurationError("Retry.INCOMPLETE needs a result probe")
        state = self._store.read()
        within, cap = None, 1
        if lease is not None:
            held = self._lease_link(state, lease)
            within, profile, slots = held.allocation_id, held.profile, slots or 1
            if slots > cast(int, held.slots):
                raise ConfigurationError(f"lease {within} holds only {held.slots} slots")
        profile = cast(Profile, profile)
        if within is None:
            cap = capacity(profile.target, profile.resources, tasks_per_allocation)
        continuation.check(profile.resources)
        observed, results = self._consult(state, request)
        self._unchanged(state)
        measured = continuation.active or within is not None
        spent = self._spent(state, observed) if measured else {}
        request = _budgeted(request, continuation, profile, spent)
        selection = decide(state, observed, results, request)

        def links(group: tuple[str, ...]) -> int:
            return max(
                len(continuation.link_limits(profile.resources, spent.get(key, _ZERO)))
                for key in group
            )

        if within is None:
            selected, deferred = batch(selection.selected, profile.target, cap, links)
        else:
            selected, deferred = selection.selected, ()
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
            continuation=continuation,
            spent={key: spent.get(key, _ZERO) for key in selected} if measured else {},
            extensions={
                key: reason for key, reason in selection.extensions.items() if key in selected
            },
            slots=slots,
            within=within,
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
        """Ask ``sbatch --test-only`` once per distinct allocation shape. Records nothing.

        A shape of a later chain link also carries ``--kill-on-invalid-dep=yes``; its
        dependency needs job numbers that exist only at submission. A step plan submits no Slurm
        job, so it has nothing to validate."""
        plan = self._verified(plan)
        if not plan.allocations or plan.decision.kind is Kind.STEP:
            return ()
        scheduler = self._scheduler(plan.decision.profile.target)
        shapes: dict[tuple[int, timedelta, bool, bool], PlannedAllocation] = {}
        for item in plan.allocations:
            later = item.link.index > 1
            relay = any(word.startswith("--signal=B:") for word in item.argv)
            shapes.setdefault((_size(item), item.time_limit, later, relay), item)
        return tuple(
            ShapeCheck(
                _size(item),
                item.cpus,
                item.memory_mib,
                item.gpus,
                item.time_limit,
                *scheduler.test_only((*item.argv, *(_LATER_LINK if later else ())), item.script),
            )
            for (_, _, later, _), item in shapes.items()
        )

    # --- Submission ---------------------------------------------------------------------------

    def submit(self, plan: Plan) -> SubmitResult:
        """Submit exactly the reviewed plan, recording durable intent before each ``sbatch``.

        The plan is rebuilt and compared first, the scheduler is pinged, and every planned Task
        must still be eligible, with the run time it was planned with; all of that happens before
        any intent. Links go in order, every chain's link 1 first, and each later link depends on
        the jobs of the links before it. A step is ordered into its lease instead of submitted,
        and its job is the lease's. A stop after that raises ``SubmissionInterrupted`` whose
        ``result`` lists receipts, unresolved allocations, and unattempted allocations.
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
                claimed = self._claim(decision, item, decision.revision + 2 * index)
            except Exception as error:
                if self._recorded(item.allocation_id):  # durable, though the commit failed
                    stop = "recording the intent failed after it became durable"
                    raise _interrupted(stop, error, receipts, unresolved, later) from error
                stop = "the next intent was not recorded"
                raise _interrupted(stop, error, receipts, (), plan.allocations[index:]) from error
            try:
                if item.within is None:
                    after = dependency_options(chain_jobs(claimed, item.link))
                    job = scheduler.submit((*item.argv, *after), item.script)
                else:
                    scheduler.order(item.within, item.allocation_id, item.script)
                    job = cast(int, claimed.attempt(item.within).job)
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
        submitted; a younger one stays unresolved and is reported as ``waiting``. A step resolves
        at once: the client claims its order, so it was not submitted when that claim won, and
        accepted when its lease had taken it.
        """
        now = self._time()
        state = self._store.read()
        unresolved = [
            item for item in state.attempts if item.acceptance is AcceptanceState.UNRESOLVED
        ]
        steps = [item for item in unresolved if item.within is not None]
        unresolved = [item for item in unresolved if item.within is None]
        rows, markers = self._evidence(unresolved, unresolved)
        outcomes: list[tuple[str, int | None]] = []
        waiting: list[WaitingAllocation] = []
        for attempt in steps:
            lease = state.attempt(cast(str, attempt.within))
            job = cast(int, lease.job)
            scheduler = self._scheduler(attempt.profile.target)
            withdrawn = scheduler.withdraw(
                lease.allocation_id, attempt.allocation_id, job, stop=False
            )
            outcomes.append((attempt.allocation_id, None if withdrawn else job))
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
        submitted. A chosen chain link brings every later link of its chain. Allocations are
        cancelled newest first, so no later link becomes eligible while an earlier one stops; after
        a failed ``scancel``, no earlier link of that chain is tried, and an unresolved link of a
        chosen chain is a ``Conflict``. Cancelling a lease ends every step in it. A step that has
        not ended is stopped inside its lease, which runs on: its order is withdrawn when the lease
        has not taken it yet, and otherwise the lease stops the step.
        Returns a receipt (with the job Slurm listed) for each allocation that was asked to stop;
        packed allocations stop all their Tasks.
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
        first: dict[str, int] = {}
        for item in chosen:
            first.setdefault(item.link.chain, item.link.index)
        chained = [
            item
            for item in state.attempts
            if item.link.chain in first and item.link.index >= first[item.link.chain]
        ]
        for item in chained:
            if item.acceptance is AcceptanceState.UNRESOLVED:
                raise Conflict(f"allocation {item.allocation_id} is unresolved; reconcile it first")
        steps = [item.allocation_id for item in chained if item.within is not None]
        rows, _ = self._evidence([item for item in chained if item.within is None])
        observed = self._observe(state, steps).attempts
        cancelled: list[Receipt] = []
        failed: list[str] = []
        broken: set[str] = set()
        skipped: list[str] = []
        for attempt in reversed(chained):
            if attempt.link.chain in broken:
                # An earlier link stopped after its pre-limit signal would hand off to the link
                # that is still queued.
                skipped.append(attempt.allocation_id)
                continue
            if attempt.within is not None:
                observation = observed.get(attempt.allocation_id)
                if observation is None or not observation.live or observation.ended:
                    continue
                job = cast(int, attempt.job)
                scheduler = self._scheduler(attempt.profile.target)
                try:
                    scheduler.withdraw(attempt.within, attempt.allocation_id, job, stop=True)
                except Unavailable as error:
                    failed.append(f"{attempt.allocation_id} ({error})")
                else:
                    cancelled.append(Receipt(attempt.allocation_id, job, attempt.task_keys))
                continue
            row = queue_row(attempt.allocation_id, rows[attempt.allocation_id], attempt.job)
            if row is None or row.settling:
                continue
            try:
                self._scheduler(attempt.profile.target).cancel(attempt.allocation_id)
            except Unavailable as error:
                failed.append(f"{attempt.allocation_id} ({error})")
                broken.add(attempt.link.chain)
            else:
                cancelled.append(Receipt(attempt.allocation_id, row.job, attempt.task_keys))
        if failed:
            done = ", ".join(item.allocation_id for item in cancelled) or "none"
            message = f"cancel failed for {'; '.join(failed)}; cancelled: {done}"
            if skipped:
                message += f"; not attempted: {', '.join(skipped)}"
            raise Unavailable(message)
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
        ones from their queue rows and run markers, not-submitted ones from their queue rows.
        A step is observed through its lease's queue row and markers plus its own markers; a
        not-submitted step that left any marker but its void is an ``EvidenceConflict``."""
        wanted = set(allocation_ids)
        chosen = [
            item
            for item in state.attempts
            if item.allocation_id in wanted and item.acceptance is not AcceptanceState.UNRESOLVED
        ]
        steps = [item for item in chosen if item.within is not None]
        leases = {item.allocation_id: item for item in map(state.attempt, _withins(steps))}
        jobs = [item for item in chosen if item.within is None]
        accepted = [item for item in jobs if item.acceptance is _ACCEPTED]
        queued = _unique((*jobs, *leases.values()))
        rows, markers = self._evidence(queued, _unique((*accepted, *steps, *leases.values())))
        observations: dict[str, Observation] = {}
        strays: dict[str, int] = {}
        for attempt in chosen:
            name = attempt.allocation_id
            if attempt.within is not None:
                lease = leases[attempt.within]
                if attempt.job is not None:
                    observations[name] = observe_step(
                        name,
                        lease.allocation_id,
                        attempt.job,
                        rows[lease.allocation_id],
                        markers[name],
                        markers[lease.allocation_id],
                    )
                elif any(not isinstance(marker, Void) for marker in markers[name]):
                    raise EvidenceConflict(f"step {name} was recorded as not submitted, yet it ran")
            elif attempt.job is not None:
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

    def _lease_link(self, state: State, lease_id: str) -> Attempt:
        """The link of the chain of lease ``lease_id`` that new steps start in: the first
        accepted one that takes orders."""
        named = state.attempt(lease_id)
        if named.kind is not Kind.LEASE:
            raise ConfigurationError(f"allocation {lease_id} is not a lease")
        chain = [
            item
            for item in state.attempts
            if item.link.chain == named.link.chain and item.acceptance is _ACCEPTED
        ]
        observed = self._observe(state, (item.allocation_id for item in chain)).attempts
        for item in chain:
            if _serving(observed[item.allocation_id]):
                return item
        stop = "has no queued or running link before its pre-limit signal"
        raise Conflict(f"lease {lease_id} {stop}")

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
        continuation, profile = decision.continuation, decision.profile
        if decision.within is not None:
            lease = self._observe(state, (decision.within,)).attempts[decision.within]
            if not _serving(lease):
                stop = f"lease {decision.within} ended or reached its pre-limit signal; plan again"
                raise StalePlan(stop)
        spent = self._spent(state, observed) if decision.measured else {}
        if decision.measured and {key: spent.get(key) for key in keys} != decision.spent:
            raise StalePlan("a planned Task's measured run time changed after planning; plan again")
        request = _budgeted(request, continuation, profile, spent)
        try:
            selection = decide(state, observed, results, request)
        except PlanRefused as error:
            raise StalePlan(f"the plan no longer holds ({error}); plan again") from error
        if selection.selected != keys:
            held = ", ".join(
                f"{key!r} ({selection.held[key]})" for key in keys if key in selection.held
            )
            raise StalePlan(f"planned Tasks are no longer eligible: {held}; plan again")
        if selection.extensions != decision.extensions:
            raise StalePlan("what the planned extensions extend changed after planning; plan again")

    def _consult(self, state: State, request: Request) -> tuple[Observed, dict[str, ResultState]]:
        """Probe and observe, once, exactly what ``decide`` may consult for ``request``."""
        tasks, allocation_ids = scope(state, request)
        results = self._results(tasks)
        return self._observe(state, allocation_ids), results

    def _claim(self, decision: Decision, item: PlannedAllocation, expected: int) -> State:
        """Record intent if the campaign is still at revision ``expected``; the committed state."""
        profile = link_profile(decision.profile, item.time_limit)

        def claim(state: State) -> State:
            if state.revision != expected:
                raise StalePlan("the campaign changed during submission; plan again")
            return record_intent(
                state,
                allocation_id=item.allocation_id,
                link=item.link,
                task_keys=item.task_keys,
                profile=profile,
                at=self._time(),
                slots=item.slots,
                within=item.within,
            )

        return self._store.update(claim)

    def _spent(self, state: State, observed: Observed) -> dict[str, timedelta]:
        """The run time of every Task whose accepted Attempts were all observed; a Task never
        accepted has run for no time."""
        now = self._time()
        spent: dict[str, timedelta] = {}
        for key, past in history(state).items():
            measured = run_time(past, observed, now)
            if measured is not None:
                spent[key] = measured.total
        return spent

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
