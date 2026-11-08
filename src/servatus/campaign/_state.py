"""Durable Campaign state: the roster, Attempt history, one invariant, and pure transitions.

Every ``State`` checks the complete invariant when constructed, so an invalid state can never be
written or read. Transitions are pure functions that return a new ``State``.
"""

from __future__ import annotations

from collections.abc import Generator, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, cast

from ..errors import ConfigurationError, Conflict, CorruptState, NotFound
from . import _codec
from ._config import MAX_LINKS, MINUTE, Profile, Task

SCHEMA_VERSION = 10


class _Invalid(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise _Invalid(message)


class Kind(StrEnum):
    """What an Attempt holds. An ``allocation`` runs its Tasks as one Slurm job. A ``lease`` is a
    Slurm job that holds ``slots`` copies of its Profile's per-Task Resources and runs no Task of
    its own. A ``step`` runs one Task in ``slots`` of the lease it is ``within``, as an ``srun``
    step of the lease's job."""

    ALLOCATION = "allocation"
    LEASE = "lease"
    STEP = "step"

    @staticmethod
    def of(slots: int | None, within: str | None) -> Kind:
        if within is not None:
            return Kind.STEP
        return Kind.ALLOCATION if slots is None else Kind.LEASE


class AcceptanceState(StrEnum):
    UNRESOLVED = "UNRESOLVED"
    ACCEPTED = "ACCEPTED"
    NOT_SUBMITTED = "NOT_SUBMITTED"


@dataclass(frozen=True, slots=True)
class RegisteredTask:
    task: Task
    revision: int


@dataclass(frozen=True, slots=True)
class Link:
    """One Attempt's place in its chain of allocations. A plain allocation is link 1 of 1.

    ``chain`` is the allocation id of link 1. ``budget`` is the lifetime run-time budget the
    chain's plan granted each of its Tasks; status shows it, and nothing decides from it.
    """

    chain: str
    index: int
    count: int
    budget: timedelta | None = None

    def __post_init__(self) -> None:
        _require(_codec.is_hex(self.chain, 24), "invalid chain identity")
        _require(
            type(self.index) is int
            and type(self.count) is int
            and 1 <= self.index <= self.count <= MAX_LINKS,
            "invalid link position",
        )
        budget = self.budget
        _require(
            budget is None or (budget > timedelta(0) and budget % MINUTE == timedelta(0)),
            "invalid link budget",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Attempt:
    """One allocation's durable record: intent before scheduler contact, then one outcome.

    The outcome is its revision plus the accepted ``job``, or ``None`` for not submitted. A lease
    has no Tasks and holds ``slots``; a step has one Task, the ``slots`` it uses, and the lease it
    runs ``within``, whose job is its job (``kind``).
    """

    allocation_id: str
    link: Link
    task_keys: tuple[str, ...]
    profile: Profile
    intent_revision: int
    intent_at: datetime
    job: int | None = None
    outcome_revision: int | None = None
    slots: int | None = None
    within: str | None = None

    def __post_init__(self) -> None:
        _require(_codec.is_hex(self.allocation_id, 24), "invalid allocation identity")
        _require(isinstance(self.link, Link), "invalid link")
        _require(
            self.link.index > 1 or self.link.chain == self.allocation_id,
            "link 1 must name its own chain",
        )
        _require(self.intent_at.utcoffset() == timedelta(0), "intent time must be UTC")
        keys, slots = self.task_keys, self.slots
        _require(len(set(keys)) == len(keys), "invalid attempt task references")
        _require(slots is None or (type(slots) is int and slots >= 1), "invalid slot count")
        _require(self.within is None or _codec.is_hex(self.within, 24), "invalid lease identity")
        kind = self.kind
        _require(
            len(keys) == 1 if kind is Kind.STEP else bool(keys) == (kind is Kind.ALLOCATION),
            f"invalid {kind} task references",
        )
        _require(
            kind is not Kind.STEP or (slots is not None and self.link.count == 1),
            "a step is one link with a slot count",
        )
        if self.outcome_revision is None:
            _require(self.job is None, "unresolved intent has a job")
        else:
            _require(self.intent_revision < self.outcome_revision, "invalid outcome revision")
        _require(
            self.job is None or (type(self.job) is int and self.job >= 1), "invalid job number"
        )

    @property
    def kind(self) -> Kind:
        return Kind.of(self.slots, self.within)

    @property
    def acceptance(self) -> AcceptanceState:
        if self.outcome_revision is None:
            return AcceptanceState.UNRESOLVED
        return AcceptanceState.NOT_SUBMITTED if self.job is None else AcceptanceState.ACCEPTED


@dataclass(frozen=True, slots=True)
class State:
    campaign_id: str
    revision: int
    roster: tuple[RegisteredTask, ...]
    sealed_revision: int | None
    attempts: tuple[Attempt, ...] = ()

    def __post_init__(self) -> None:
        _check(self)

    @property
    def tasks(self) -> tuple[Task, ...]:
        return tuple(item.task for item in self.roster)

    @property
    def sealed(self) -> bool:
        return self.sealed_revision is not None

    def attempt(self, allocation_id: str) -> Attempt:
        for attempt in self.attempts:
            if attempt.allocation_id == allocation_id:
                return attempt
        raise NotFound(f"unknown allocation: {allocation_id}")


@dataclass(frozen=True, slots=True)
class TaskHistory:
    """One Task's Attempts in state order: accepted ones (with the Task's slot in each), ones
    recorded as not submitted, and the unresolved one, which the invariant makes the last."""

    accepted: tuple[tuple[Attempt, int], ...] = ()
    not_submitted: tuple[Attempt, ...] = ()
    unresolved: Attempt | None = None


def history(state: State) -> dict[str, TaskHistory]:
    """Every roster Task's history, from one pass over the Attempts."""
    accepted: dict[str, list[tuple[Attempt, int]]] = {}
    not_submitted: dict[str, list[Attempt]] = {}
    unresolved: dict[str, Attempt] = {}
    for attempt in state.attempts:
        acceptance = attempt.acceptance
        for slot, key in enumerate(attempt.task_keys):
            if acceptance is AcceptanceState.ACCEPTED:
                accepted.setdefault(key, []).append((attempt, slot))
            elif acceptance is AcceptanceState.NOT_SUBMITTED:
                not_submitted.setdefault(key, []).append(attempt)
            else:
                unresolved[key] = attempt
    return {
        task.key: TaskHistory(
            tuple(accepted.get(task.key, ())),
            tuple(not_submitted.get(task.key, ())),
            unresolved.get(task.key),
        )
        for task in state.tasks
    }


def _check(state: State) -> None:
    """The single Campaign invariant, linear in tasks plus attempted Task references."""
    revision, sealed = state.revision, state.sealed_revision
    _require(_codec.is_hex(state.campaign_id, 32), "invalid campaign identity")
    _require(revision >= 0, "invalid revision")
    _require(sealed is None or 0 <= sealed <= revision, "invalid seal revision")
    events: set[int] = set()
    position: dict[str, int] = {}
    introduced: dict[str, int] = {}
    previous = 0
    for index, item in enumerate(state.roster):
        when = item.revision
        _require(previous <= when <= revision, "invalid task revision")
        _require(sealed is None or when <= sealed, "task appended after seal")
        _require(item.task.key not in position, "duplicate task key")
        position[item.task.key], introduced[item.task.key], previous = index, when, when
        if when:
            events.add(when)
    if sealed:
        _require(sealed not in events, "conflicting seal revision")
        events.add(sealed)

    def event(value: int) -> None:
        _require(value <= revision and value not in events, "invalid event revision")
        events.add(value)

    identities: set[str] = set()
    links: dict[tuple[str, int], Attempt] = {}
    leases: dict[str, Attempt] = {}
    busy_until: dict[str, float] = {}
    last_intent = 0
    for attempt in state.attempts:
        intent, keys, link = attempt.intent_revision, attempt.task_keys, attempt.link
        _require(attempt.allocation_id not in identities, "duplicate allocation identity")
        identities.add(attempt.allocation_id)
        _require((link.chain, link.index) not in links, "duplicate chain link")
        if link.index > 1:
            previous = links.get((link.chain, link.index - 1))
            _require(
                previous is not None
                and previous.link.count == link.count
                and previous.link.budget == link.budget
                and previous.task_keys == keys
                and previous.slots == attempt.slots
                and previous.job is not None
                and cast(int, previous.outcome_revision) < intent,
                "a link must follow the accepted previous link of its chain",
            )
        links[link.chain, link.index] = attempt
        if attempt.within is not None:
            _check_step(attempt, leases.get(attempt.within))
        if attempt.kind is Kind.LEASE:
            leases[attempt.allocation_id] = attempt
        _require(intent > last_intent, "invalid intent revision")
        last_intent = intent
        event(intent)
        if attempt.outcome_revision is not None:
            event(attempt.outcome_revision)
        last = -1
        for name in keys:
            _require(
                name in position and introduced[name] < intent, "invalid attempt task references"
            )
            _require(position[name] > last, "attempt tasks are not in roster order")
            last = position[name]
            _require(busy_until.get(name, 0) < intent, "attempt overlaps unresolved intent")
        until = float("inf") if attempt.outcome_revision is None else attempt.outcome_revision
        for name in keys:
            busy_until[name] = max(busy_until.get(name, 0), until)
    _require(revision == max(events, default=0), "revision does not match recorded mutations")


def _check_step(step: Attempt, lease: Attempt | None) -> None:
    """A step runs in an earlier lease accepted before its intent, on the lease's Profile and
    within its slots; once accepted, its job is the lease's job."""
    _require(
        lease is not None
        and lease.job is not None
        and cast(int, lease.outcome_revision) < step.intent_revision,
        "a step must run in a lease accepted before its intent",
    )
    lease = cast(Attempt, lease)
    _require(
        step.profile == lease.profile and cast(int, step.slots) <= cast(int, lease.slots),
        "a step must fit its lease's Profile and slots",
    )
    _require(step.job is None or step.job == lease.job, "a step's job must be its lease's job")


# --- Pure transitions ------------------------------------------------------------------------


@contextmanager
def _rejected() -> Generator[None, None, None]:
    """Report an invariant violation from a transition as ``Conflict``."""
    try:
        yield
    except _Invalid as error:
        raise Conflict(f"campaign state change rejected: {error}") from None


def _changed(state: State, **changes: object) -> State:
    with _rejected():
        return replace(state, **changes)  # pyright: ignore[reportArgumentType]


def _unique_tasks(tasks: Iterable[Task]) -> tuple[Task, ...]:
    frozen = tuple(tasks)
    if any(not isinstance(task, Task) for task in frozen):
        raise ConfigurationError("tasks must contain Task values")
    if len({task.key for task in frozen}) != len(frozen):
        raise ConfigurationError("Task keys must be unique")
    return frozen


def create(campaign_id: str, tasks: Iterable[Task], *, appendable: bool) -> State:
    if type(appendable) is not bool:
        raise ConfigurationError("appendable must be a bool")
    roster = tuple(RegisteredTask(task, 0) for task in _unique_tasks(tasks))
    with _rejected():
        return State(campaign_id, 0, roster, None if appendable else 0)


def append(state: State, tasks: Iterable[Task]) -> State:
    suffix = _unique_tasks(tasks)
    if state.sealed:
        raise Conflict("a sealed campaign roster cannot change")
    known = {task.key for task in state.tasks}
    if clashes := [task.key for task in suffix if task.key in known]:
        raise Conflict(f"appended Task keys already exist: {', '.join(clashes)}")
    if not suffix:
        return state
    revision = state.revision + 1
    roster = state.roster + tuple(RegisteredTask(task, revision) for task in suffix)
    return _changed(state, revision=revision, roster=roster)


def seal(state: State) -> State:
    if state.sealed:
        return state
    return _changed(state, revision=state.revision + 1, sealed_revision=state.revision + 1)


def record_intent(
    state: State,
    *,
    allocation_id: str,
    link: Link,
    task_keys: tuple[str, ...],
    profile: Profile,
    at: datetime,
    slots: int | None = None,
    within: str | None = None,
) -> State:
    """Record intent for one allocation, lease, or step (``Kind``). A link after the first must
    follow the accepted previous link of its chain, with the same Tasks; a step must fit an
    accepted lease; otherwise ``Conflict``."""
    with _rejected():
        attempt = Attempt(
            allocation_id=allocation_id,
            link=link,
            task_keys=task_keys,
            profile=profile,
            intent_revision=state.revision + 1,
            intent_at=at,
            slots=slots,
            within=within,
        )
    return _changed(state, revision=attempt.intent_revision, attempts=state.attempts + (attempt,))


def chain_jobs(state: State, link: Link) -> tuple[int, ...]:
    """The jobs of the links before ``link`` in its chain, in order. The invariant guarantees
    that each was accepted before ``link``'s intent."""
    jobs = {
        attempt.link.index: attempt.job
        for attempt in state.attempts
        if attempt.link.chain == link.chain and attempt.link.index < link.index
    }
    return tuple(cast(int, jobs[index]) for index in range(1, link.index))


def record_outcome(state: State, allocation_id: str, job: int | None) -> State:
    """Record acceptance of ``job`` or non-submission (``None``) at the next revision. Identical
    repeats are idempotent; a conflicting outcome raises ``Conflict``."""
    attempt = state.attempt(allocation_id)
    if attempt.outcome_revision is not None:
        if attempt.job == job:
            return state
        raise Conflict(f"allocation {allocation_id} already has a conflicting outcome")
    revision = state.revision + 1
    with _rejected():
        resolved = replace(attempt, job=job, outcome_revision=revision)
    attempts = tuple(resolved if item is attempt else item for item in state.attempts)
    return _changed(state, revision=revision, attempts=attempts)


# --- Document (schema 10) ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class _AttemptDocument:
    allocation_id: str
    link: Link
    task_keys: tuple[str, ...]
    profile: str
    intent_revision: int
    intent_at: datetime
    job: int | None
    outcome_revision: int | None
    slots: int | None
    within: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class _StateDocument:
    schema_version: int
    campaign_id: str
    revision: int
    sealed_revision: int | None
    tasks: tuple[RegisteredTask, ...]
    profiles: Mapping[str, Profile]
    attempts: tuple[_AttemptDocument, ...]


def _fields(value: Attempt | _AttemptDocument, **changes: object) -> dict[str, Any]:
    """An Attempt or its document as keyword arguments for the other, with ``changes``."""
    return {item.name: getattr(value, item.name) for item in fields(value)} | changes


def _profile_key(profile: Profile) -> str:
    return _codec.digest(_codec.dump(profile))[:16]


def encode(state: State) -> bytes:
    profiles = {_profile_key(attempt.profile): attempt.profile for attempt in state.attempts}
    document = _StateDocument(
        schema_version=SCHEMA_VERSION,
        campaign_id=state.campaign_id,
        revision=state.revision,
        sealed_revision=state.sealed_revision,
        tasks=state.roster,
        profiles=profiles,
        attempts=tuple(
            _AttemptDocument(**_fields(attempt, profile=_profile_key(attempt.profile)))
            for attempt in state.attempts
        ),
    )
    try:
        return _codec.canonical(_codec.dump(document)) + b"\n"
    except ValueError as error:
        raise ConfigurationError(str(error)) from None


def decode(data: bytes) -> State:
    """Decode and fully validate a state document; any defect raises ``CorruptState``."""
    try:
        raw = _codec.decode_json(data)
        if (
            not isinstance(raw, dict)
            or cast(dict[str, object], raw).get("schema_version") != SCHEMA_VERSION
        ):
            raise _Invalid(f"unsupported campaign schema; expected {SCHEMA_VERSION}")
        document = _codec.load(_StateDocument, cast(object, raw))
        profiles = dict(document.profiles)
        for name, profile in profiles.items():
            _require(_profile_key(profile) == name, "profile key does not match its content")
        used = {item.profile for item in document.attempts}
        _require(used == profiles.keys(), "profiles do not match attempt references")
        state = State(
            campaign_id=document.campaign_id,
            revision=document.revision,
            roster=document.tasks,
            sealed_revision=document.sealed_revision,
            attempts=tuple(
                Attempt(**_fields(item, profile=profiles[item.profile]))
                for item in document.attempts
            ),
        )
        canonical = encode(state) == data
    except (ValueError, TypeError, ConfigurationError) as error:
        raise CorruptState(f"invalid campaign state: {error}") from None
    if not canonical:
        raise CorruptState("campaign state is not canonically encoded")
    return state
