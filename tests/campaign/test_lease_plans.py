"""Lease and step plans built from a Decision against Campaign state, without a scheduler."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from support.builders import profile, resources, target, tasks

from servatus.campaign._config import Continuation, Profile
from servatus.campaign._plan import Decision, Plan, build_plan, decode_decision, slot_resources
from servatus.campaign._policy import Hold
from servatus.campaign._script import Chained, render_lease, render_order, sbatch_argv
from servatus.campaign._state import Kind, Link, State, create, record_intent, record_outcome
from servatus.errors import ConfigurationError

NOW = datetime(2030, 1, 1, tzinfo=UTC)
KEYS: tuple[str, ...] = ("task-0", "task-1", "task-2")
HELD = resources(
    cpus=4, memory_mib=1024, gpus=1, time_limit="01:00:00", signal_before_end="00:05:00"
)
LEASED = profile(resource_value=HELD)
LEASE_JOB = 500
NOT_A_LEASE = "a step plan must fit an accepted lease of the campaign"
ONE_EACH = "a lease or step plan needs a slot count and one Task each"


def identity(number: int) -> str:
    return f"{number:024x}"


def attempt(
    state: State,
    keys: tuple[str, ...] = (),
    *,
    slots: int | None = None,
    job: int | None = LEASE_JOB,
    resolved: bool = True,
    profile_value: Profile = LEASED,
) -> tuple[State, str]:
    name = identity(len(state.attempts) + 1)
    state = record_intent(
        state,
        allocation_id=name,
        link=Link(name, 1, 1),
        task_keys=keys,
        profile=profile_value,
        at=NOW,
        slots=slots,
    )
    return (record_outcome(state, name, job) if resolved else state), name


def leased(profile_value: Profile = LEASED, slots: int = 4) -> tuple[State, str]:
    """Three Tasks and one accepted lease of ``slots``."""
    return attempt(
        create("c" * 32, tasks(3), appendable=False), slots=slots, profile_value=profile_value
    )


def decision(state: State, **changes: Any) -> Decision:
    selected = changes.get("selected", ())
    deferred = changes.get("deferred", ())
    values: dict[str, Any] = {
        "campaign_id": state.campaign_id,
        "revision": state.revision,
        "nonce": "0" * 32,
        "profile": LEASED,
        "selected": selected,
        "held": {key: Hold.NOT_REQUESTED for key in KEYS if key not in (*selected, *deferred)},
        "deferred": deferred,
        "tasks_per_allocation": 1,
        "probe_required": False,
        "continuation": Continuation(),
        "spent": {},
        "extensions": {},
        "slots": 4,
    }
    values.update(changes)
    return Decision(**values)


def steps(state: State, lease: str, **changes: Any) -> Decision:
    selected = changes.pop("selected", ("task-0", "task-2"))
    values: dict[str, Any] = {
        "selected": selected,
        "spent": dict.fromkeys(selected, timedelta(0)),
        "slots": 1,
        "within": lease,
    }
    return decision(state, **(values | changes))


# --- Leases ----------------------------------------------------------------------------------


def test_a_lease_decision_builds_a_chain_of_leases_holding_its_slots() -> None:
    state, _ = leased()
    lease = decision(state, slots=2, continuation=Continuation(max_links=3))
    plan = build_plan(state, lease)
    ids = [item.allocation_id for item in plan.allocations]
    assert len(set(ids)) == 3 and ids == [
        item.allocation_id for item in build_plan(state, lease).allocations
    ]
    other = build_plan(state, decision(state, slots=2, nonce="1" * 32))
    assert other.allocations[0].allocation_id not in ids
    assert plan.decision.kind is Kind.LEASE and plan.retry == ()
    for index, item in enumerate(plan.allocations, start=1):
        assert item.link == Link(ids[0], index, 3)
        assert (item.task_keys, item.slots, item.within) == ((), 2, None)
        assert (item.cpus, item.memory_mib, item.gpus) == (8, 2048, 2)
        assert item.time_limit == timedelta(hours=1)
        assert item.argv == sbatch_argv(LEASED.target, HELD, 2, item.allocation_id, relay=True)
        assert "--signal=B:USR1@300" in item.argv and "--ntasks=2" in item.argv
        assert item.script == render_lease(LEASED.target, HELD, item.allocation_id)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"selected": ("task-0",)}, "a lease plan selects no Task and takes no budget"),
        ({"continuation": Continuation(budget="03:00:00")}, "selects no Task and takes no budget"),
        ({"tasks_per_allocation": 2}, ONE_EACH),
        ({"slots": 0}, ONE_EACH),
    ],
)
def test_impossible_lease_decisions_are_refused(changes: dict[str, Any], message: str) -> None:
    state, _ = leased()
    with pytest.raises(ConfigurationError, match=message):
        build_plan(state, decision(state, **changes))


# --- Steps -----------------------------------------------------------------------------------


def test_a_step_decision_builds_one_order_per_selected_task_in_the_lease() -> None:
    state, lease = leased()
    budget = Continuation(budget="02:00:00")
    spent = {"task-0": timedelta(seconds=90), "task-2": timedelta(0)}
    plan = build_plan(state, steps(state, lease, slots=2, continuation=budget, spent=spent))
    assert plan.decision.kind is Kind.STEP and plan.decision.measured
    assert [item.task_keys for item in plan.allocations] == [("task-0",), ("task-2",)]
    scaled = slot_resources(HELD, 2)
    roster = dict(zip(KEYS, tasks(3), strict=True))
    for item in plan.allocations:
        (key,) = item.task_keys
        assert item.link == Link(item.allocation_id, 1, 1, timedelta(hours=2))
        assert (item.argv, item.slots, item.within) == ((), 2, lease)
        assert (item.cpus, item.memory_mib, item.gpus) == (8, 2048, 2)
        assert item.time_limit == timedelta(hours=1)
        chain = Chained(1, timedelta(hours=2), (1,), (spent[key],))
        expected = render_order(LEASED.target, scaled, roster[key], item.allocation_id, chain)
        assert item.script == expected
    assert len({item.allocation_id for item in plan.allocations} | {lease}) == 3


def test_steps_always_record_their_run_time() -> None:
    state, lease = leased()
    with pytest.raises(ConfigurationError, match="plan run times do not match its selected Tasks"):
        build_plan(state, steps(state, lease, spent={"task-0": timedelta(0)}))


def unresolved() -> tuple[State, str]:
    return attempt(create("c" * 32, tasks(3), appendable=False), slots=4, resolved=False)


def not_submitted() -> tuple[State, str]:
    return attempt(create("c" * 32, tasks(3), appendable=False), slots=4, job=None)


def allocation() -> tuple[State, str]:
    state, _ = leased()
    return attempt(state, ("task-1",), job=LEASE_JOB + 1)


@pytest.mark.parametrize(
    ("build", "changes", "message"),
    [
        (leased, {"within": identity(99)}, NOT_A_LEASE),
        (unresolved, {}, NOT_A_LEASE),
        (not_submitted, {}, NOT_A_LEASE),
        (allocation, {}, NOT_A_LEASE),
        (leased, {"profile": profile(label="other", resource_value=HELD)}, NOT_A_LEASE),
        (leased, {"slots": 5}, NOT_A_LEASE),
        (leased, {"slots": None}, ONE_EACH),
        (leased, {"tasks_per_allocation": 2}, ONE_EACH),
        (
            leased,
            {"selected": ("task-0",), "deferred": ("task-2",)},
            "steps are one link each and never deferred",
        ),
        (
            leased,
            {"continuation": Continuation(max_links=2)},
            "steps are one link each and never deferred",
        ),
    ],
    ids=[
        "unknown",
        "unresolved",
        "not-submitted",
        "allocation",
        "other-profile",
        "too-many-slots",
        "no-slots",
        "packed",
        "deferred",
        "max-links",
    ],
)
def test_impossible_step_decisions_are_refused(
    build: Callable[[], tuple[State, str]], changes: dict[str, Any], message: str
) -> None:
    state, within = build()
    if "deferred" in changes:
        changes["spent"] = {"task-0": timedelta(0)}
    with pytest.raises(ConfigurationError, match=message):
        build_plan(state, steps(state, within, **changes))


def test_steps_ignore_the_submit_cap_that_lease_links_count_against() -> None:
    capped = profile(target(max_allocations_per_submit=1), HELD)
    state, lease = leased(capped)
    plan = build_plan(state, steps(state, lease, profile=capped, selected=KEYS))
    assert len(plan.allocations) == 3
    links = decision(state, profile=capped, continuation=Continuation(max_links=2))
    with pytest.raises(ConfigurationError, match="exceeds the target's max_allocations_per_submit"):
        build_plan(state, links)


# --- Saved plans and warnings ----------------------------------------------------------------


def built(kind: str) -> tuple[State, Plan]:
    state, lease = leased()
    if kind == "lease":
        return state, build_plan(state, decision(state, slots=3))
    return state, build_plan(state, steps(state, lease, slots=2))


@pytest.mark.parametrize("kind", ["lease", "step"])
def test_lease_and_step_plans_round_trip_as_plan_4(kind: str) -> None:
    state, plan = built(kind)
    data = plan.to_json()
    document = json.loads(data)
    assert document["format"] == "servatus.plan/4"
    assert (document["slots"], document["within"]) == (
        plan.decision.slots,
        plan.decision.within,
    )
    restored, digest = decode_decision(data)
    assert (restored, digest) == (plan.decision, plan.digest)
    assert build_plan(state, restored).digest == plan.digest


def test_lease_and_step_plans_carry_no_budget_notes() -> None:
    state, lease = leased()
    short = Continuation(budget="01:03:00")
    zero = dict.fromkeys(KEYS, timedelta(0))
    plain = decision(
        state, selected=KEYS, slots=None, continuation=short, spent=zero, profile=LEASED
    )
    assert any("dropped" in note for note in build_plan(state, plain).warnings)
    stepped = steps(state, lease, selected=KEYS, continuation=short)
    assert build_plan(state, stepped).warnings == ()
    assert build_plan(state, decision(state)).warnings == ()
