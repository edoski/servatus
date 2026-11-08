"""Pure planning: the reviewable ``Decision``, allocation capacity and packing, and ``Plan``.

A ``Decision`` is everything an operator reviews: which Tasks a plan submits, why every other
Task is held or deferred, the Profile it uses, and how far it continues them (``Continuation``,
with each selected Task's measured run time). ``build_plan`` derives the rest (chains of
allocations, their identities, batch scripts, ``sbatch`` argv, and the digest) from a Decision and
the current Campaign state, so a saved plan stays compact (no Task arguments or stdin) and a
submitted plan can always be rebuilt and compared with what was reviewed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import timedelta
from types import MappingProxyType
from typing import cast

from .._fs import StrPath
from ..errors import ConfigurationError, StalePlan
from ..publication import publish_file
from . import _codec
from ._codec import format_duration
from ._config import MAX_LINKS, MINUTE, Continuation, Profile, Resources, Target, Task
from ._policy import Extension, Hold
from ._script import Chained, check_command, dependency_options, render_batch, sbatch_argv
from ._state import Link, State, history

PLAN_FORMAT = "servatus.plan/3"
_ZERO = timedelta(0)
# The longest job number Slurm prints, to bound a link's dependency option at planning.
_LONGEST_JOB = 4_294_967_295


@dataclass(frozen=True, slots=True, kw_only=True)
class Decision:
    """The reviewed part of a plan. Every other plan field is derived from it.

    ``selected`` Tasks are packed into this plan's allocations; ``deferred`` Tasks were eligible
    but exceed the target's ``max_allocations_per_submit``; every other Task is ``held`` with
    one reason. ``nonce`` makes every plan's allocation identities unique, even for copies of
    the same Campaign state. ``continuation`` says how far the plan continues its Tasks;
    ``spent`` is each selected Task's run time, measured at planning, when it is active; and
    ``extensions`` names the selected Tasks that extend finished work under a budget.
    """

    campaign_id: str
    revision: int
    nonce: str
    profile: Profile
    selected: tuple[str, ...]
    held: Mapping[str, Hold]
    deferred: tuple[str, ...]
    tasks_per_allocation: int
    probe_required: bool
    continuation: Continuation
    spent: Mapping[str, timedelta]
    extensions: Mapping[str, Extension]

    def __post_init__(self) -> None:
        for name in ("held", "spent", "extensions"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


@dataclass(frozen=True, slots=True)
class PlannedAllocation:
    """One single-node allocation: its place in its chain, its Tasks, total request, this link's
    time limit, and exact ``sbatch`` input. A later link's ``argv`` lacks the dependency on the
    earlier links' jobs, which submission adds once they are known."""

    allocation_id: str
    link: Link
    task_keys: tuple[str, ...]
    cpus: int
    memory_mib: int
    gpus: int
    time_limit: timedelta
    script: bytes = field(repr=False)
    argv: tuple[str, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class Plan:
    """A reviewed ``decision`` plus what it derives: allocations (every chain's link 1, then every
    link 2, and so on), the digest, and ``retry``, the selected or deferred Tasks that resubmit
    accepted work.

    Scripts and argv contain Task arguments, environment, and stdin: treat them as sensitive.
    ``to_json`` and ``save`` write only the Decision and digest.
    """

    decision: Decision
    allocations: tuple[PlannedAllocation, ...]
    digest: str
    retry: tuple[str, ...]

    @property
    def extensions(self) -> tuple[str, ...]:
        """The selected Tasks that extend finished work, in roster order."""
        return tuple(key for key in self.decision.selected if key in self.decision.extensions)

    @property
    def warnings(self) -> tuple[str, ...]:
        notes: list[str] = []
        if deferred := self.decision.deferred:
            notes.append(
                f"{len(deferred)} eligible Tasks are deferred by max_allocations_per_submit; "
                "plan again after submitting"
            )
        return (*notes, *_budget_notes(self.decision))

    def to_json(self) -> bytes:
        """Canonical JSON: ``{"format": "servatus.plan/3"}``, the Decision, and the digest."""
        document = cast(dict[str, object], _codec.dump(self.decision))
        document["format"] = PLAN_FORMAT
        document["digest"] = self.digest
        return _codec.canonical(document)

    def save(self, path: StrPath) -> None:
        """Write ``to_json`` as a new owner-only (0600) file; never overwrites."""
        data = self.to_json()
        publish_file(path, lambda stage: stage.write_bytes(data), mode=0o600)


def decode_decision(data: bytes) -> tuple[Decision, str]:
    """Parse a saved plan document into its Decision and recorded digest."""
    if not isinstance(data, bytes):
        raise ConfigurationError("a plan document must be bytes")
    try:
        raw = _codec.decode_json(data)
        if not isinstance(raw, dict):
            raise _codec.CodecError("expected a JSON object")
        document = dict(cast(dict[str, object], raw))
        if document.pop("format", None) != PLAN_FORMAT:
            raise _codec.CodecError(f"expected format {PLAN_FORMAT!r}")
        digest = document.pop("digest", None)
        if not _codec.is_hex(digest, 64):
            raise _codec.CodecError("expected a 64-digit hexadecimal digest")
        return _codec.load(Decision, document), digest
    except (ValueError, TypeError, ConfigurationError) as error:
        raise ConfigurationError(f"invalid plan document: {error}") from None


def capacity(target: Target, resources: Resources, cap: int | None = None) -> int:
    """Tasks per allocation: the most that fit every target ceiling, or the requested ``cap``.

    Raises ``ConfigurationError`` when one Task cannot fit, GPU work has no ``gpu_gres``, or
    ``cap`` exceeds what fits.
    """
    if not isinstance(target, Target) or not isinstance(resources, Resources):
        raise ConfigurationError("capacity needs a Target and Resources")
    if resources.time_limit > target.max_time_limit:
        raise ConfigurationError("time_limit exceeds the target's max_time_limit")
    if resources.gpus and target.gpu_gres is None:
        raise ConfigurationError("GPU work needs a target gpu_gres")
    ceilings = {
        "max_tasks_per_allocation": target.max_tasks_per_allocation,
        "max_cpus_per_allocation": target.max_cpus_per_allocation // resources.cpus,
        "max_memory_mib_per_allocation": (
            target.max_memory_mib_per_allocation // resources.memory_mib
        ),
    }
    if resources.gpus:
        ceilings["max_gpus_per_allocation"] = target.max_gpus_per_allocation // resources.gpus
    name, feasible = min(ceilings.items(), key=lambda item: item[1])
    if feasible < 1:
        raise ConfigurationError(f"one Task exceeds the target's {name}")
    if cap is None:
        return feasible
    if type(cap) is not int or cap < 1:
        raise ConfigurationError("tasks_per_allocation must be a positive integer")
    if cap > feasible:
        raise ConfigurationError(
            f"tasks_per_allocation {cap} exceeds the feasible capacity {feasible} ({name})"
        )
    return cap


def batch(
    keys: tuple[str, ...],
    target: Target,
    cap: int,
    links: Callable[[tuple[str, ...]], int] = lambda _: 1,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``keys`` split into the Tasks one submit carries and the deferred rest: the longest prefix
    whose packed groups of ``cap`` need at most ``max_allocations_per_submit`` allocations, with
    each group's chain counting its ``links``. Whole chains are deferred, never cut short."""
    limit = target.max_allocations_per_submit
    if limit is None:
        return keys, ()
    for size in range(min(len(keys), cap * limit), 0, -1):
        if sum(links(group) for group in pack(keys[:size], cap)) <= limit:
            return keys[:size], keys[size:]
    return (), keys


def pack(keys: Sequence[str], cap: int) -> tuple[tuple[str, ...], ...]:
    """Split ``keys`` in order into the fewest groups of at most ``cap``, sizes differing by
    at most one (larger groups first)."""
    if not keys:
        return ()
    count = -(-len(keys) // cap)
    small, larger = divmod(len(keys), count)
    groups: list[tuple[str, ...]] = []
    offset = 0
    for index in range(count):
        size = small + (index < larger)
        groups.append(tuple(keys[offset : offset + size]))
        offset += size
    return tuple(groups)


@dataclass(frozen=True, slots=True)
class _Chain:
    """One packed group's chain: its Task keys and each slot's own link time limits."""

    keys: tuple[str, ...]
    limits: tuple[tuple[timedelta, ...], ...]

    @property
    def count(self) -> int:
        return max(len(own) for own in self.limits)

    def limit(self, index: int) -> timedelta:
        """Link ``index``'s time limit: the longest any slot still running there needs."""
        return max(own[index - 1] for own in self.limits if len(own) >= index)


def _chains(decision: Decision) -> tuple[_Chain, ...]:
    continuation, resources = decision.continuation, decision.profile.resources
    return tuple(
        _Chain(
            keys,
            tuple(
                continuation.link_limits(resources, decision.spent.get(key, _ZERO)) for key in keys
            ),
        )
        for keys in pack(decision.selected, decision.tasks_per_allocation)
    )


def _budget_notes(decision: Decision) -> list[str]:
    """What a budgeted plan cannot spend as asked: rests too short for a link, budgets beyond
    ``max_links``, and packed Tasks whose chains need different link counts."""
    continuation, resources = decision.continuation, decision.profile.resources
    if continuation.budget is None:
        return []
    reserve = continuation.reserve(resources)
    cap = continuation.max_links or MAX_LINKS
    dropped: list[str] = []
    unreachable: list[str] = []
    notes: list[str] = []
    for chain in _chains(decision):
        for key, own in zip(chain.keys, chain.limits, strict=True):
            left = continuation.budget - decision.spent.get(key, _ZERO) - sum(own, _ZERO)
            if left >= MINUTE:
                (unreachable if len(own) == cap else dropped).append(key)
        if len(set(chain.limits)) > 1:
            shown = ", ".join(
                f"{key} {format_duration(sum(own, _ZERO))} in {len(own)}"
                for key, own in zip(chain.keys, chain.limits, strict=True)
            )
            notes.append(
                f"Tasks packed together need different links ({shown}); a Task runs to its "
                "link's limit, which a needier Task may have set, unless it stops itself at "
                "SERVATUS_BUDGET, and later links hold resources for Tasks that ended; plan with "
                "--tasks-per-allocation 1 to avoid it"
            )
    if unreachable:
        notes.insert(
            0,
            f"{cap} links cannot spend the whole budget of "
            f"{', '.join(unreachable)}; plan again once they finish",
        )
    if dropped:
        notes.insert(
            0,
            f"the last minutes of the budget of {', '.join(dropped)} are dropped: a link must "
            f"outlast signal_before_end ({format_duration(reserve - timedelta(minutes=1))}) "
            "by more than a minute",
        )
    return notes


def link_profile(profile: Profile, time_limit: timedelta) -> Profile:
    """``profile`` with one link's time limit."""
    resources = profile.resources
    if resources.time_limit == time_limit:
        return profile
    return replace(
        profile,
        resources=Resources(
            cpus=resources.cpus,
            memory_mib=resources.memory_mib,
            time_limit=time_limit,
            gpus=resources.gpus,
            signal_before_end=resources.signal_before_end,
        ),
    )


def _check(state: State, decision: Decision) -> None:
    if decision.campaign_id != state.campaign_id:
        raise StalePlan("the plan belongs to a different campaign")
    if decision.revision != state.revision:
        raise StalePlan(
            f"the campaign changed after planning (revision {decision.revision} is now "
            f"{state.revision}); plan again"
        )
    roster = tuple(task.key for task in state.tasks)
    parts = (decision.selected, tuple(decision.held), decision.deferred)
    if sorted(key for part in parts for key in part) != sorted(roster):
        raise ConfigurationError("plan Tasks do not partition the campaign roster")
    position = {key: index for index, key in enumerate(roster)}
    for part in (decision.selected, decision.deferred):
        if list(part) != sorted(part, key=position.__getitem__):
            raise ConfigurationError("plan Task keys are not in roster order")
    continuation = decision.continuation
    continuation.check(decision.profile.resources)
    measured: set[str] = set(decision.selected) if continuation.active else set()
    if set(decision.spent) != measured:
        raise ConfigurationError("plan run times do not match its selected Tasks")
    if not set(decision.extensions) <= set(decision.selected) or (
        decision.extensions and continuation.budget is None
    ):
        raise ConfigurationError("plan extensions must be selected Tasks under a budget")


def build_plan(state: State, decision: Decision) -> Plan:
    """Derive chains, allocations, scripts, argv, and the digest from ``decision`` against
    ``state``.

    Each packed group of selected Tasks is one chain with as many links as its neediest Task;
    a plain plan's chains have one link. ``StalePlan`` when the decision was made for another
    campaign or revision; ``ConfigurationError`` when it is inconsistent or a rendered request
    breaks a bound.
    """
    _check(state, decision)
    held = decision.held
    decision = replace(
        decision, held={task.key: held[task.key] for task in state.tasks if task.key in held}
    )
    profile, continuation = decision.profile, decision.continuation
    target = profile.target
    capacity(target, profile.resources, decision.tasks_per_allocation)
    document = _codec.dump(decision)
    by_key = {task.key: task for task in state.tasks}
    chains = _chains(decision)
    if any(not own for chain in chains for own in chain.limits):
        raise ConfigurationError("the plan selects a Task whose budget is already spent")
    ids = [
        [
            _codec.digest({"decision": document, "index": index, "link": link})[:24]
            for link in range(1, chain.count + 1)
        ]
        for index, chain in enumerate(chains)
    ]
    allocations: list[PlannedAllocation] = []
    for link in range(1, max((chain.count for chain in chains), default=0) + 1):
        for chain, own in zip(chains, ids, strict=True):
            if link > chain.count:
                continue
            allocation_id = own[link - 1]
            resources = link_profile(profile, chain.limit(link)).resources
            chained = None
            if continuation.active:
                chained = Chained(
                    link,
                    continuation.budget,
                    tuple(len(limits) for limits in chain.limits),
                    tuple(decision.spent[key] for key in chain.keys),
                    own[link] if link < chain.count else None,
                )
            group: tuple[Task, ...] = tuple(by_key[key] for key in chain.keys)
            script = render_batch(target, resources, group, allocation_id, chained)
            argv = sbatch_argv(target, resources, len(group), allocation_id, chained)
            check_command((*argv, *dependency_options((_LONGEST_JOB,) * (link - 1)), "--test-only"))
            allocations.append(
                PlannedAllocation(
                    allocation_id=allocation_id,
                    link=Link(own[0], link, chain.count, continuation.budget),
                    task_keys=chain.keys,
                    cpus=len(group) * resources.cpus,
                    memory_mib=len(group) * resources.memory_mib,
                    gpus=len(group) * resources.gpus,
                    time_limit=resources.time_limit,
                    script=script,
                    argv=argv,
                )
            )
    limit = target.max_allocations_per_submit
    if limit is not None and len(allocations) > limit:
        raise ConfigurationError("plan exceeds the target's max_allocations_per_submit")
    scripts = [hashlib.sha256(item.script).hexdigest() for item in allocations]
    digest = _codec.digest({"decision": document, "scripts": scripts})
    past = history(state)
    retry = tuple(key for key in decision.selected + decision.deferred if past[key].accepted)
    return Plan(decision, tuple(allocations), digest, retry)
