"""Durable Campaign state: the roster, Attempt history, one invariant, and pure transitions.

Every ``State`` checks the complete invariant when constructed, so an invalid state can never be
written or read. Transitions are pure functions that return a new ``State``.
"""

from __future__ import annotations

import re
from collections.abc import Generator, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, cast

from ..errors import ConfigurationError, Conflict, CorruptState, NotFound
from . import _codec
from ._codec import FLATTEN
from ._config import Profile, Task
from ._evidence import JobRef

SCHEMA_VERSION = 7
_HEX24 = re.compile(r"[0-9a-f]{24}\Z")
_HEX32 = re.compile(r"[0-9a-f]{32}\Z")


class _Invalid(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise _Invalid(message)


def _subsequence(part: tuple[str, ...], whole: tuple[str, ...]) -> bool:
    remaining = iter(whole)
    return all(any(item == other for other in remaining) for item in part)


class AcceptanceState(StrEnum):
    UNRESOLVED = "UNRESOLVED"
    ACCEPTED = "ACCEPTED"
    NOT_SUBMITTED = "NOT_SUBMITTED"


@dataclass(frozen=True, slots=True)
class RegisteredTask:
    task: Task = field(metadata=FLATTEN)
    revision: int

    def __post_init__(self) -> None:
        _require(type(self.revision) is int and self.revision >= 0, "invalid task revision")


@dataclass(frozen=True, slots=True, kw_only=True)
class Attempt:
    """One allocation's durable record: intent before scheduler contact, then one outcome."""

    allocation_id: str
    task_keys: tuple[str, ...]
    profile: Profile
    retry: tuple[str, ...]
    duplicate_risk: tuple[str, ...]
    intent_revision: int
    intent_at: datetime
    acceptance: AcceptanceState = AcceptanceState.UNRESOLVED
    job: JobRef | None = None
    outcome_revision: int | None = None

    def __post_init__(self) -> None:
        _require(_HEX24.fullmatch(self.allocation_id) is not None, "invalid allocation identity")
        _require(self.intent_revision >= 1, "invalid intent revision")
        _require(self.intent_at.utcoffset() == timedelta(0), "intent time must be UTC")
        keys = self.task_keys
        _require(bool(keys) and len(set(keys)) == len(keys), "invalid attempt task references")
        _require(_subsequence(self.retry, keys), "invalid retry references")
        _require(_subsequence(self.duplicate_risk, self.retry), "invalid duplicate-risk references")
        if self.acceptance is AcceptanceState.UNRESOLVED:
            _require(
                self.outcome_revision is None and self.job is None, "unresolved intent has outcome"
            )
        else:
            _require(
                self.outcome_revision is not None and self.intent_revision < self.outcome_revision,
                "invalid outcome revision",
            )
            _require(
                (self.job is not None) == (self.acceptance is AcceptanceState.ACCEPTED),
                "job identity disagrees with acceptance",
            )
        target, resources, count = self.profile.target, self.profile.resources, len(keys)
        _require(
            count <= target.max_tasks_per_allocation
            and count * resources.cpus <= target.max_cpus_per_allocation
            and count * resources.memory_mib <= target.max_memory_mib_per_allocation
            and count * resources.gpus <= target.max_gpus_per_allocation
            and resources.time_limit <= target.max_time_limit,
            "attempt exceeds its target capacity",
        )


@dataclass(frozen=True, slots=True)
class State:
    campaign_id: str
    revision: int
    roster: tuple[RegisteredTask, ...]
    sealed_revision: int | None
    attempts: tuple[Attempt, ...] = ()

    def __post_init__(self) -> None:
        check(self)

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


def check(state: State) -> None:
    """The single Campaign invariant, linear in tasks plus attempted Task references."""
    revision, sealed = state.revision, state.sealed_revision
    _require(_HEX32.fullmatch(state.campaign_id) is not None, "invalid campaign identity")
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
    busy_until: dict[str, float] = {}
    accepted: set[str] = set()
    last_intent = 0
    for attempt in state.attempts:
        intent, keys = attempt.intent_revision, attempt.task_keys
        _require(attempt.allocation_id not in identities, "duplicate allocation identity")
        identities.add(attempt.allocation_id)
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
        _require(
            tuple(name for name in keys if name in accepted) == attempt.retry,
            "retry choices disagree with prior accepted work",
        )
        until = float("inf") if attempt.outcome_revision is None else attempt.outcome_revision
        for name in keys:
            busy_until[name] = max(busy_until.get(name, 0), until)
        if attempt.acceptance is AcceptanceState.ACCEPTED:
            accepted.update(keys)
    _require(revision == max(events, default=0), "revision does not match recorded mutations")


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
    task_keys: tuple[str, ...],
    profile: Profile,
    retry: tuple[str, ...],
    duplicate_risk: tuple[str, ...],
    at: datetime,
) -> tuple[State, Attempt]:
    with _rejected():
        attempt = Attempt(
            allocation_id=allocation_id,
            task_keys=task_keys,
            profile=profile,
            retry=retry,
            duplicate_risk=duplicate_risk,
            intent_revision=state.revision + 1,
            intent_at=at,
        )
    changed = _changed(
        state, revision=attempt.intent_revision, attempts=state.attempts + (attempt,)
    )
    return changed, attempt


def record_outcome(state: State, allocation_id: str, job: JobRef | None) -> State:
    """Record acceptance (``job``) or explicit non-submission (``None``). Identical repeats are
    idempotent; a conflicting outcome raises ``Conflict``."""
    attempt = state.attempt(allocation_id)
    acceptance = AcceptanceState.NOT_SUBMITTED if job is None else AcceptanceState.ACCEPTED
    if attempt.acceptance is not AcceptanceState.UNRESOLVED:
        if attempt.acceptance is acceptance and attempt.job == job:
            return state
        raise Conflict(f"allocation {allocation_id} already has a conflicting outcome")
    return _resolved(state, attempt, job)


def correct_outcome(state: State, allocation_id: str, job: JobRef) -> State:
    """Replace a NOT_SUBMITTED outcome with acceptance of ``job`` at a new revision.

    Only for scheduler proof that the allocation was accepted after all. Identical repeats are
    idempotent. ``Conflict`` when the Attempt is not recorded as not submitted, or when a later
    Attempt already names one of its Tasks (that history cannot be rewritten).
    """
    attempt = state.attempt(allocation_id)
    if attempt.acceptance is AcceptanceState.ACCEPTED and attempt.job == job:
        return state
    if attempt.acceptance is not AcceptanceState.NOT_SUBMITTED:
        raise Conflict(f"allocation {allocation_id} is not recorded as not submitted")
    keys = set(attempt.task_keys)
    later = state.attempts[state.attempts.index(attempt) + 1 :]
    if clashes := [item.allocation_id for item in later if keys.intersection(item.task_keys)]:
        raise Conflict(
            f"allocation {allocation_id} cannot be marked accepted: later allocations "
            f"{', '.join(clashes)} already include its Tasks"
        )
    return _resolved(state, attempt, job)


def _resolved(state: State, attempt: Attempt, job: JobRef | None) -> State:
    """``state`` with ``attempt``'s outcome recorded at the next revision."""
    acceptance = AcceptanceState.NOT_SUBMITTED if job is None else AcceptanceState.ACCEPTED
    revision = state.revision + 1
    with _rejected():
        resolved = replace(attempt, acceptance=acceptance, job=job, outcome_revision=revision)
    attempts = tuple(resolved if item is attempt else item for item in state.attempts)
    return _changed(state, revision=revision, attempts=attempts)


# --- Document (schema 7) ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class _AttemptDocument:
    allocation_id: str
    task_keys: tuple[str, ...]
    profile: str
    retry: tuple[str, ...]
    duplicate_risk: tuple[str, ...]
    intent_revision: int
    intent_at: datetime
    acceptance: AcceptanceState
    job: JobRef | None
    outcome_revision: int | None


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


def profile_key(profile: Profile) -> str:
    return _codec.digest(_codec.dump(profile))[:16]


def encode(state: State) -> bytes:
    profiles = {profile_key(attempt.profile): attempt.profile for attempt in state.attempts}
    document = _StateDocument(
        schema_version=SCHEMA_VERSION,
        campaign_id=state.campaign_id,
        revision=state.revision,
        sealed_revision=state.sealed_revision,
        tasks=state.roster,
        profiles=profiles,
        attempts=tuple(
            _AttemptDocument(**_fields(attempt, profile=profile_key(attempt.profile)))
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
            _require(profile_key(profile) == name, "profile key does not match its content")
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
