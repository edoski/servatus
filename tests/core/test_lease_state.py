"""Leases and steps in durable state: their Attempt shapes, the invariant, and the document."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from core_helpers import NOW, allocation_id, intend, reencode, roster, submit
from support.builders import profile

from servatus.campaign._state import (
    Kind,
    Link,
    State,
    decode,
    encode,
    history,
    record_intent,
    record_outcome,
)
from servatus.errors import Conflict, CorruptState

OTHER = profile(label="other")
STEP_REFUSED = "a step must run in a lease accepted before its intent"
UNFIT = "a step must use its lease's Profile"
ONE_SLOT = "a step uses one slot of its lease"


def leased(slots: int = 4) -> tuple[State, str]:
    """Three Tasks and an accepted lease of ``slots``."""
    return submit(roster(3), [], slots=slots)


def job_of(state: State, identity: str) -> int:
    job = state.attempt(identity).job
    assert job is not None
    return job


def _intent(state: State, **changes: Any) -> State:
    identity = allocation_id(len(state.attempts) + 1)
    values: dict[str, Any] = {
        "allocation_id": identity,
        "link": Link(identity, 1, 1),
        "task_keys": ("task-0",),
        "profile": profile(),
        "at": NOW,
    }
    values.update(changes)
    return record_intent(state, **values)


@pytest.mark.parametrize(
    ("slots", "within", "kind"),
    [
        (None, None, Kind.ALLOCATION),
        (4, None, Kind.LEASE),
        (None, "a" * 24, Kind.STEP),
    ],
)
def test_slots_and_within_name_the_kind(slots: int | None, within: str | None, kind: Kind) -> None:
    assert Kind.of(slots, within) is kind


def test_a_lease_holds_slots_and_a_step_runs_in_it() -> None:
    state, lease = leased()
    state, step = submit(state, ["task-0"], within=lease, job=job_of(state, lease))
    assert state.attempt(lease).kind is Kind.LEASE and state.attempt(lease).task_keys == ()
    assert state.attempt(step).kind is Kind.STEP
    assert (state.attempt(step).slots, state.attempt(step).within) == (None, lease)


# --- Attempt shapes --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"slots": 2}, "invalid lease task references"),
        ({"task_keys": ()}, "invalid allocation task references"),
        ({"task_keys": (), "within": "LEASE"}, "invalid step task references"),
        ({"task_keys": ("task-0", "task-1"), "within": "LEASE"}, "invalid step task references"),
        ({"slots": 1, "within": "LEASE"}, ONE_SLOT),
        ({"within": "LEASE", "link": Link(allocation_id(2), 1, 2)}, "a step is one link"),
        ({"task_keys": (), "slots": 0}, "invalid slot count"),
        ({"task_keys": (), "slots": True}, "invalid slot count"),
        ({"within": "X" * 24}, "invalid lease identity"),
    ],
)
def test_impossible_lease_and_step_attempts_are_conflicts(
    changes: dict[str, Any], message: str
) -> None:
    state, lease = leased()
    if changes.get("within") == "LEASE":
        changes["within"] = lease
    with pytest.raises(Conflict, match=message):
        _intent(state, **changes)


# --- The invariant ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda: (roster(3), allocation_id(99)),
        lambda: submit(roster(3), ["task-1"]),
        lambda: intend(roster(3), [], slots=4),
        lambda: submit(roster(3), [], slots=4, accepted=False),
    ],
    ids=["unknown", "allocation", "unresolved", "not-submitted"],
)
def test_a_step_needs_an_accepted_lease(build: Callable[[], tuple[State, str]]) -> None:
    state, within = build()
    with pytest.raises(Conflict, match=STEP_REFUSED):
        intend(state, ["task-0"], within=within)


def test_a_step_uses_its_lease_profile() -> None:
    state, lease = leased(4)
    with pytest.raises(Conflict, match=UNFIT):
        intend(state, ["task-0"], within=lease, profile_value=OTHER)
    assert intend(state, ["task-0"], within=lease)[0].attempts[-1].within == lease


def test_a_step_accepts_only_its_lease_job() -> None:
    state, lease = leased()
    state, step = intend(state, ["task-0"], within=lease)
    with pytest.raises(Conflict, match="a step's job must be its lease's job"):
        record_outcome(state, step, job_of(state, lease) + 1)
    assert record_outcome(state, step, job_of(state, lease)).attempt(step).job == job_of(
        state, lease
    )
    assert record_outcome(state, step, None).attempt(step).job is None


def test_a_lease_link_keeps_the_slots_of_its_chain() -> None:
    first = allocation_id(1)
    state, _ = submit(roster(1), [], slots=4, link=Link(first, 1, 3))
    with pytest.raises(Conflict, match="must follow the accepted previous link"):
        intend(state, [], slots=2, link=Link(first, 2, 3))
    state, second = intend(state, [], slots=4, link=Link(first, 2, 3))
    assert state.attempt(second).kind is Kind.LEASE


def test_history_ignores_leases_and_puts_a_step_task_in_slot_0() -> None:
    state, lease = leased()
    state, step = submit(state, ["task-1"], within=lease, job=job_of(state, lease))
    past = history(state)
    assert past["task-1"].accepted == ((state.attempt(step), 0),)
    assert all(not own.accepted for key, own in past.items() if key != "task-1")


# --- The document ----------------------------------------------------------------------------


def documented() -> State:
    """An allocation on another Profile, an accepted lease of 4, and a step in it."""
    state, _ = submit(roster(3), ["task-1"], profile_value=OTHER)
    state, lease = submit(state, [], slots=4)
    state, _ = submit(state, ["task-0"], within=lease, job=job_of(state, lease))
    return state


def test_leases_and_steps_round_trip_with_their_slots_and_lease() -> None:
    state = documented()
    data = encode(state)
    assert decode(data) == state
    raw = json.loads(data)
    lease = raw["attempts"][1]["allocation_id"]
    assert [(item["slots"], item["within"]) for item in raw["attempts"]] == [
        (None, None),
        (4, None),
        (None, lease),
    ]
    assert raw["attempts"][1]["task_keys"] == [] and raw["attempts"][2]["job"] == 102


def _late_lease(raw: dict[str, Any]) -> None:
    raw["attempts"][1]["outcome_revision"] = 5
    raw["attempts"][2]["intent_revision"] = 4


EDITS: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
    (STEP_REFUSED, lambda r: r["attempts"][2].update(within="f" * 24)),
    (STEP_REFUSED, lambda r: r["attempts"][2].update(within=r["attempts"][0]["allocation_id"])),
    (STEP_REFUSED, _late_lease),
    (UNFIT, lambda r: r["attempts"][2].update(profile=r["attempts"][0]["profile"])),
    (ONE_SLOT, lambda r: r["attempts"][2].update(slots=1)),
    ("a step's job must be its lease's job", lambda r: r["attempts"][2].update(job=7)),
    ("a step is one link", lambda r: r["attempts"][2]["link"].update(count=2)),
    (
        "invalid step task references",
        lambda r: r["attempts"][2].update(task_keys=["task-0", "task-2"]),
    ),
    ("invalid lease task references", lambda r: r["attempts"][1].update(task_keys=["task-2"])),
    ("invalid slot count", lambda r: r["attempts"][1].update(slots=0)),
    ("invalid lease identity", lambda r: r["attempts"][2].update(within="lease")),
]


@pytest.mark.parametrize(("message", "edit"), EDITS, ids=[f"{i}" for i in range(len(EDITS))])
def test_a_document_breaking_a_lease_rule_is_corrupt_state(
    message: str, edit: Callable[[dict[str, Any]], None]
) -> None:
    raw = json.loads(encode(documented()))
    edit(raw)
    with pytest.raises(CorruptState, match=f"invalid campaign state: .*{message}"):
        decode(reencode(raw))
