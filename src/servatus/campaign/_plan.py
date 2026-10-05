"""Pure planning: the reviewable ``Decision``, allocation capacity and packing, and ``Plan``.

A ``Decision`` is everything an operator reviews: which Tasks a plan submits, why every other
Task is held or deferred, and the Profile it uses. ``build_plan`` derives the rest (allocation
identities, batch scripts, ``sbatch`` argv, and the digest) from a Decision and the current
Campaign state, so a saved plan stays compact (no Task arguments or stdin) and a submitted plan
can always be rebuilt and compared with what was reviewed.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import timedelta
from types import MappingProxyType
from typing import cast

from ..errors import ConfigurationError, StalePlan
from ..publication import publish_file
from . import _codec
from ._config import Profile, Resources, StrPath, Target, Task
from ._policy import Hold
from ._remote import check_command
from ._script import render_batch, sbatch_argv
from ._state import AcceptanceState, State

PLAN_FORMAT = "servatus.plan/1"
_HEX32 = re.compile(r"[0-9a-f]{32}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _keys(value: tuple[str, ...], name: str) -> None:
    if any(not isinstance(key, str) for key in value) or len(set(value)) != len(value):
        raise ConfigurationError(f"plan {name} must be distinct Task keys")


@dataclass(frozen=True, slots=True, kw_only=True)
class Decision:
    """The reviewed part of a plan. Every other plan field is derived from it.

    ``selected`` Tasks are packed into this plan's allocations; ``deferred`` Tasks were eligible
    but exceed the target's ``max_allocations_per_submit``; every other Task is ``held`` with
    one reason. ``retry`` and ``duplicate_risk`` list the selected or deferred Tasks that
    resubmit accepted work (with acknowledged duplicate risk). ``nonce`` makes every plan's
    allocation identities unique, even for copies of the same Campaign state.
    """

    campaign_id: str
    revision: int
    nonce: str
    profile: Profile
    selected: tuple[str, ...]
    held: Mapping[str, Hold]
    deferred: tuple[str, ...]
    retry: tuple[str, ...]
    duplicate_risk: tuple[str, ...]
    tasks_per_allocation: int
    probe_required: bool

    def __post_init__(self) -> None:
        if not isinstance(self.profile, Profile):
            raise ConfigurationError("plan profile must be a Profile")
        if _HEX32.fullmatch(self.nonce) is None or _HEX32.fullmatch(self.campaign_id) is None:
            raise ConfigurationError("plan identities must be 32 lowercase hexadecimal digits")
        if type(self.revision) is not int or self.revision < 0:
            raise ConfigurationError("plan revision must be a non-negative integer")
        if type(self.tasks_per_allocation) is not int or self.tasks_per_allocation < 1:
            raise ConfigurationError("tasks_per_allocation must be a positive integer")
        if type(self.probe_required) is not bool:
            raise ConfigurationError("probe_required must be a bool")
        for name in ("selected", "deferred", "retry", "duplicate_risk"):
            _keys(cast(tuple[str, ...], getattr(self, name)), name)
        held = dict(self.held)
        if any(
            not isinstance(key, str) or not isinstance(hold, Hold) for key, hold in held.items()
        ):
            raise ConfigurationError("plan held Tasks must map Task keys to Hold reasons")
        object.__setattr__(self, "held", MappingProxyType(held))


@dataclass(frozen=True, slots=True)
class PlannedAllocation:
    """One single-node allocation: its Tasks, total request, and exact ``sbatch`` input."""

    allocation_id: str
    task_keys: tuple[str, ...]
    cpus: int
    memory_mib: int
    gpus: int
    time_limit: timedelta
    script: bytes = field(repr=False)
    argv: tuple[str, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class Plan:
    """A reviewed ``Decision`` plus its derived allocations and digest.

    Scripts and argv contain Task arguments, environment, and stdin: treat them as sensitive.
    ``to_json`` and ``save`` write only the Decision and digest.
    """

    decision: Decision
    allocations: tuple[PlannedAllocation, ...]
    digest: str

    @property
    def campaign_id(self) -> str:
        return self.decision.campaign_id

    @property
    def revision(self) -> int:
        return self.decision.revision

    @property
    def profile(self) -> Profile:
        return self.decision.profile

    @property
    def selected(self) -> tuple[str, ...]:
        return self.decision.selected

    @property
    def held(self) -> Mapping[str, Hold]:
        return self.decision.held

    @property
    def deferred(self) -> tuple[str, ...]:
        return self.decision.deferred

    @property
    def retry(self) -> tuple[str, ...]:
        return self.decision.retry

    @property
    def duplicate_risk(self) -> tuple[str, ...]:
        return self.decision.duplicate_risk

    @property
    def tasks_per_allocation(self) -> int:
        return self.decision.tasks_per_allocation

    @property
    def probe_required(self) -> bool:
        return self.decision.probe_required

    @property
    def warnings(self) -> tuple[str, ...]:
        notes: list[str] = []
        if self.duplicate_risk:
            keys = ", ".join(self.duplicate_risk)
            notes.append(f"duplicate execution risk acknowledged for: {keys}")
        if self.deferred:
            notes.append(
                f"{len(self.deferred)} eligible Tasks are deferred by max_allocations_per_submit; "
                "plan again after submitting"
            )
        return tuple(notes)

    def to_json(self) -> bytes:
        """Canonical JSON: ``{"format": "servatus.plan/1"}``, the Decision, and the digest."""
        document = cast(dict[str, object], _codec.dump(self.decision))
        document["format"] = PLAN_FORMAT
        document["digest"] = self.digest
        return _codec.canonical(document) + b"\n"

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
        if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
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
    eligible = decision.selected + decision.deferred
    accepted = {
        key
        for attempt in state.attempts
        if attempt.acceptance is AcceptanceState.ACCEPTED
        for key in attempt.task_keys
    }
    if set(decision.retry) != accepted.intersection(eligible):
        raise ConfigurationError("plan retry keys disagree with accepted Campaign history")
    if not set(decision.duplicate_risk) <= set(decision.retry):
        raise ConfigurationError("plan duplicate-risk keys must be retried keys")
    position = {key: index for index, key in enumerate(roster)}
    for part in (decision.selected, decision.deferred, decision.retry, decision.duplicate_risk):
        if list(part) != sorted(part, key=position.__getitem__):
            raise ConfigurationError("plan Task keys are not in roster order")
    limit = decision.profile.target.max_allocations_per_submit
    cap = decision.tasks_per_allocation
    if limit is not None and len(decision.selected) > cap * limit:
        raise ConfigurationError("plan exceeds the target's max_allocations_per_submit")


def build_plan(state: State, decision: Decision) -> Plan:
    """Derive allocations, scripts, argv, and the digest from ``decision`` against ``state``.

    ``StalePlan`` when the decision was made for another campaign or revision;
    ``ConfigurationError`` when it is inconsistent or a rendered request breaks a bound.
    """
    _check(state, decision)
    held = decision.held
    decision = replace(
        decision, held={task.key: held[task.key] for task in state.tasks if task.key in held}
    )
    profile = decision.profile
    target, resources = profile.target, profile.resources
    cap = capacity(target, resources, decision.tasks_per_allocation)
    document = _codec.dump(decision)
    by_key = {task.key: task for task in state.tasks}
    allocations: list[PlannedAllocation] = []
    for index, keys in enumerate(pack(decision.selected, cap)):
        allocation_id = _codec.digest({"decision": document, "index": index})[:24]
        group: tuple[Task, ...] = tuple(by_key[key] for key in keys)
        script = render_batch(target, resources, group, allocation_id)
        argv = sbatch_argv(target, resources, len(group), allocation_id)
        check_command((*argv, "--test-only"))
        allocations.append(
            PlannedAllocation(
                allocation_id=allocation_id,
                task_keys=keys,
                cpus=len(keys) * resources.cpus,
                memory_mib=len(keys) * resources.memory_mib,
                gpus=len(keys) * resources.gpus,
                time_limit=resources.time_limit,
                script=script,
                argv=argv,
            )
        )
    scripts = [hashlib.sha256(item.script).hexdigest() for item in allocations]
    digest = _codec.digest({"decision": document, "scripts": scripts})
    return Plan(decision, tuple(allocations), digest)
