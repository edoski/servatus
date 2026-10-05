from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import pytest
from core_helpers import CAMPAIGN_ID, NOW, allocation_id, intend, reencode, roster, submit
from support.builders import profile, target, tasks

from servatus.campaign import _codec
from servatus.campaign._config import Profile, Task
from servatus.campaign._evidence import JobRef
from servatus.campaign._state import (
    SCHEMA_VERSION,
    AcceptanceState,
    Attempt,
    RegisteredTask,
    State,
    append,
    correct_outcome,
    create,
    decode,
    encode,
    record_intent,
    record_outcome,
    seal,
)
from servatus.errors import ConfigurationError, Conflict, CorruptState, NotFound

# --- transitions -----------------------------------------------------------------------------


def test_create_registers_an_ordered_roster_sealed_unless_appendable() -> None:
    fixed = create(CAMPAIGN_ID, tasks(2), appendable=False)
    assert fixed.tasks == tasks(2) and fixed.sealed and fixed.sealed_revision == 0
    assert fixed.revision == 0 and fixed.attempts == ()
    growing = create(CAMPAIGN_ID, tasks(2), appendable=True)
    assert not growing.sealed and growing.sealed_revision is None


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (
            lambda: create(CAMPAIGN_ID, tasks(1), appendable=1),  # pyright: ignore[reportArgumentType]
            ConfigurationError,
            "bool",
        ),
        (
            lambda: create(CAMPAIGN_ID, tasks(1) * 2, appendable=False),
            ConfigurationError,
            "Task keys must be unique",
        ),
        (
            lambda: create(CAMPAIGN_ID, ["a"], appendable=False),  # pyright: ignore[reportArgumentType]
            ConfigurationError,
            "Task values",
        ),
        (lambda: create("bad", tasks(1), appendable=False), Conflict, "campaign identity"),
    ],
)
def test_create_rejects_bad_inputs(
    call: Callable[[], object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        call()


def test_append_accepts_only_a_new_suffix_and_seal_is_irreversible() -> None:
    state = roster(1, appendable=True)
    assert append(state, ()) is state
    grown = append(state, tasks(3)[1:])
    assert grown.tasks == tasks(3) and grown.revision == 1
    assert [item.revision for item in grown.roster] == [0, 1, 1]
    with pytest.raises(Conflict, match="already exist: task-0"):
        append(grown, tasks(1))
    with pytest.raises(ConfigurationError, match="unique"):
        append(grown, [Task("x"), Task("x")])
    sealed = seal(grown)
    assert sealed.sealed_revision == 2 == sealed.revision
    assert seal(sealed) is sealed
    with pytest.raises(Conflict, match="sealed campaign roster cannot change"):
        append(sealed, [Task("late")])


def test_intent_then_outcome_records_chronology_and_retry_choices() -> None:
    state, first = submit(roster(2), ["task-0", "task-1"])
    attempt = state.attempt(first)
    assert attempt.acceptance is AcceptanceState.ACCEPTED
    assert (attempt.intent_revision, attempt.outcome_revision, state.revision) == (1, 2, 2)
    state, second = intend(state, ["task-1"], ack=["task-1"])
    retried = state.attempt(second)
    assert retried.retry == ("task-1",) and retried.duplicate_risk == ("task-1",)
    assert retried.acceptance is AcceptanceState.UNRESOLVED and retried.job is None


def test_outcome_is_idempotent_and_conflicts_are_rejected() -> None:
    state, identity = intend(roster(1), ["task-0"])
    accepted = record_outcome(state, identity, JobRef(42, "alpha"))
    assert record_outcome(accepted, identity, JobRef(42, "alpha")) is accepted
    with pytest.raises(Conflict, match="conflicting outcome"):
        record_outcome(accepted, identity, JobRef(43))
    with pytest.raises(Conflict, match="conflicting outcome"):
        record_outcome(accepted, identity, None)
    rejected = record_outcome(state, identity, None)
    assert rejected.attempt(identity).acceptance is AcceptanceState.NOT_SUBMITTED
    assert record_outcome(rejected, identity, None) is rejected
    with pytest.raises(Conflict, match="conflicting outcome"):
        record_outcome(rejected, identity, JobRef(42))
    with pytest.raises(NotFound, match="unknown allocation"):
        record_outcome(state, allocation_id(99), None)
    with pytest.raises(NotFound, match="unknown allocation"):
        state.attempt(allocation_id(99))


def test_a_not_submitted_outcome_is_corrected_at_a_new_revision() -> None:
    state, identity = submit(roster(2, appendable=True), ["task-0"], job=None)
    state = append(state, [Task("late")])  # unrelated history in between is fine
    corrected = correct_outcome(state, identity, JobRef(42))
    attempt = corrected.attempt(identity)
    assert attempt.acceptance is AcceptanceState.ACCEPTED and attempt.job == JobRef(42)
    assert attempt.outcome_revision == corrected.revision == state.revision + 1
    assert decode(encode(corrected)) == corrected
    assert correct_outcome(corrected, identity, JobRef(42)) is corrected
    with pytest.raises(Conflict, match="is not recorded as not submitted"):
        correct_outcome(corrected, identity, JobRef(43))
    pending, unresolved = intend(roster(1), ["task-0"])
    with pytest.raises(Conflict, match="is not recorded as not submitted"):
        correct_outcome(pending, unresolved, JobRef(42))
    with pytest.raises(NotFound, match="unknown allocation"):
        correct_outcome(state, allocation_id(99), JobRef(42))


def test_a_not_submitted_outcome_followed_by_its_tasks_cannot_be_corrected() -> None:
    state, identity = submit(roster(2), ["task-0", "task-1"], job=None)
    state, _ = submit(state, ["task-0"])
    state, later = intend(state, ["task-1"])
    with pytest.raises(Conflict, match=f"later allocations .*, {later} already include"):
        correct_outcome(state, identity, JobRef(42))


def test_outcome_survives_unrelated_authoring() -> None:
    state, identity = intend(roster(1, appendable=True), ["task-0"])
    state = seal(append(state, [Task("late")]))
    resolved = record_outcome(state, identity, JobRef(7))
    assert resolved.attempt(identity).outcome_revision == 4 == resolved.revision


def _intent(state: State, **changes: Any) -> State:
    values: dict[str, Any] = {
        "allocation_id": allocation_id(len(state.attempts) + 1),
        "task_keys": ("task-0",),
        "profile": profile(),
        "retry": (),
        "duplicate_risk": (),
        "at": NOW,
    }
    values.update(changes)
    return record_intent(state, **values)[0]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"allocation_id": "NOT-HEX"}, "invalid allocation identity"),
        ({"task_keys": ()}, "invalid attempt task references"),
        ({"task_keys": ("task-0", "task-0")}, "invalid attempt task references"),
        ({"task_keys": ("foreign",)}, "invalid attempt task references"),
        ({"task_keys": ("task-1", "task-0")}, "not in roster order"),
        ({"retry": ("task-0",)}, "retry choices disagree"),
        ({"retry": ("task-1",)}, "invalid retry references"),
        ({"duplicate_risk": ("task-0",)}, "invalid duplicate-risk references"),
        (
            {"task_keys": tuple(f"task-{i}" for i in range(5))},
            "attempt exceeds its target capacity",
        ),
        ({"profile": profile(target(max_time_limit="01:00:00"))}, "exceeds its target capacity"),
    ],
)
def test_record_intent_rejects_invalid_attempts_as_conflicts(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(Conflict, match=message):
        _intent(roster(5), **changes)


def test_record_intent_rejects_overlap_with_unresolved_work_and_missing_retry() -> None:
    state, _ = intend(roster(2), ["task-0"])
    with pytest.raises(Conflict, match="overlaps unresolved intent"):
        _intent(state, task_keys=("task-0", "task-1"))
    accepted, _ = submit(roster(2), ["task-0"])
    with pytest.raises(Conflict, match="retry choices disagree"):
        _intent(accepted, task_keys=("task-0", "task-1"))
    assert _intent(accepted, task_keys=("task-0", "task-1"), retry=("task-0",)).revision == 3


# --- documents -------------------------------------------------------------------------------


def history() -> State:
    """Accepted, retried, appended, sealed, and trailing unresolved work."""
    state, _ = submit(roster(3, appendable=True), ["task-0", "task-1"])
    state = append(state, [Task("task-3", ("x",), env={"A": "1"})])
    state, _ = submit(state, ["task-0", "task-2"], ack=["task-0"])
    state = seal(state)
    state, _ = submit(state, ["task-1"], job=None)
    state, _ = intend(state, ["task-3"])
    return state


def test_documents_round_trip_canonically() -> None:
    state = history()
    data = encode(state)
    assert data.endswith(b"\n") and data == reencode(json.loads(data))
    assert decode(data) == state
    raw = json.loads(data)
    assert raw["schema_version"] == SCHEMA_VERSION == 7
    assert [attempt["acceptance"] for attempt in raw["attempts"]] == [
        "ACCEPTED",
        "ACCEPTED",
        "NOT_SUBMITTED",
        "UNRESOLVED",
    ]
    assert len(raw["profiles"]) == 1
    assert {attempt["profile"] for attempt in raw["attempts"]} == set(raw["profiles"])


def test_distinct_profiles_are_stored_once_each() -> None:
    other = profile(label="other")
    state, _ = submit(roster(2), ["task-0"])
    state, _ = submit(state, ["task-1"], profile_value=other)
    state, _ = submit(state, ["task-0"])
    raw = json.loads(encode(state))
    assert len(raw["profiles"]) == 2
    assert decode(encode(state)).attempts[1].profile == other


def _edit(edit: Callable[[dict[str, Any]], None]) -> bytes:
    raw = json.loads(encode(history()))
    edit(raw)
    return reencode(raw)


def _set_profile(raw: dict[str, Any], **target_changes: Any) -> None:
    ((name, value),) = raw["profiles"].items()
    value["target"].update(target_changes)
    new = _codec.digest(value)[:16]
    raw["profiles"] = {new: value}
    for attempt in raw["attempts"]:
        attempt["profile"] = new
    assert name != new


def _unresolve_first(raw: dict[str, Any]) -> None:
    raw["attempts"][0].update(acceptance="UNRESOLVED", job=None, outcome_revision=None)


EDITS: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
    ("invalid campaign identity", lambda r: r.update(campaign_id="C" * 32)),
    ("invalid revision", lambda r: r.update(revision=-1)),
    ("invalid seal revision", lambda r: r.update(sealed_revision=99)),
    ("invalid task revision", lambda r: r["tasks"][0].update(revision=3)),
    ("invalid task revision", lambda r: r["tasks"][0].update(revision=-1)),
    ("task appended after seal", lambda r: r.update(sealed_revision=2)),
    ("duplicate task key", lambda r: r["tasks"][2].update(key="task-0")),
    ("conflicting seal revision", lambda r: r.update(sealed_revision=3)),
    ("invalid event revision", lambda r: r["attempts"][0].update(outcome_revision=3)),
    ("invalid event revision", lambda r: r["attempts"][0].update(outcome_revision=99)),
    (
        "duplicate allocation identity",
        lambda r: r["attempts"][1].update(allocation_id=r["attempts"][0]["allocation_id"]),
    ),
    ("invalid intent revision", lambda r: r["attempts"][1].update(intent_revision=1)),
    ("invalid intent revision", lambda r: r["attempts"][0].update(intent_revision=0)),
    (
        "invalid attempt task references",
        lambda r: r["attempts"][0].update(task_keys=["task-0", "task-3"]),
    ),
    ("invalid attempt task references", lambda r: r["attempts"][0].update(task_keys=[])),
    (
        "attempt tasks are not in roster order",
        lambda r: r["attempts"][1].update(task_keys=["task-2", "task-0"]),
    ),
    ("attempt overlaps unresolved intent", _unresolve_first),
    (
        "retry choices disagree with prior accepted work",
        lambda r: r["attempts"][1].update(retry=[], duplicate_risk=[]),
    ),
    ("revision does not match recorded mutations", lambda r: r.update(revision=99)),
    ("invalid allocation identity", lambda r: r["attempts"][0].update(allocation_id="x" * 24)),
    ("invalid retry references", lambda r: r["attempts"][1].update(retry=["task-3"])),
    (
        "invalid duplicate-risk references",
        lambda r: r["attempts"][1].update(duplicate_risk=["task-2"]),
    ),
    ("unresolved intent has outcome", lambda r: r["attempts"][3].update(outcome_revision=9)),
    (
        "unresolved intent has outcome",
        lambda r: r["attempts"][3].update(job={"job_id": 1, "cluster": None}),
    ),
    ("invalid outcome revision", lambda r: r["attempts"][0].update(outcome_revision=1)),
    ("invalid outcome revision", lambda r: r["attempts"][0].update(outcome_revision=None)),
    (
        "job identity disagrees with acceptance",
        lambda r: r["attempts"][0].update(acceptance="NOT_SUBMITTED"),
    ),
    (
        "job identity disagrees with acceptance",
        lambda r: r["attempts"][2].update(acceptance="ACCEPTED"),
    ),
    ("attempt exceeds its target capacity", lambda r: _set_profile(r, max_tasks_per_allocation=1)),
    ("unsupported campaign schema; expected 7", lambda r: r.update(schema_version=6)),
    ("unsupported campaign schema", lambda r: r.pop("schema_version")),
    ("expected int", lambda r: r.update(schema_version=7.0)),
    ("profile key does not match its content", lambda r: _rename_profile(r)),
    ("profiles do not match attempt references", lambda r: _add_unused_profile(r)),
    ("expected int", lambda r: r.update(revision=True)),
    ("expected int", lambda r: r["tasks"][0].update(revision=False)),
    ("unknown keys", lambda r: r["attempts"][0].update(extra=1)),
    ("unknown keys", lambda r: r.update(extra=1)),
    ("missing key 'env'", lambda r: r["tasks"][0].pop("env")),
    ("expected base64", lambda r: r["tasks"][0].update(stdin="!")),
    ("Task.env names must be identifiers", lambda r: r["tasks"][0].update(env={"1A": "x"})),
    ("cannot start with SERVATUS_", lambda r: r["tasks"][0].update(env={"SERVATUS_X": "1"})),
    ("Task.key cannot contain NUL", lambda r: r["tasks"][0].update(key="a\0")),
    ("job_id must be a positive integer", lambda r: r["attempts"][0]["job"].update(job_id=0)),
    ("cluster must be one safe site token", lambda r: r["attempts"][0]["job"].update(cluster="")),
    ("canonical UTC datetime", lambda r: r["attempts"][0].update(intent_at="2030-01-01T00:00:00")),
    ("expected one of", lambda r: r["attempts"][0].update(acceptance="MAYBE")),
]


def _rename_profile(raw: dict[str, Any]) -> None:
    (value,) = raw["profiles"].values()
    raw["profiles"] = {"0" * 16: value}
    for attempt in raw["attempts"]:
        attempt["profile"] = "0" * 16


def _add_unused_profile(raw: dict[str, Any]) -> None:
    (value,) = raw["profiles"].values()
    other = json.loads(json.dumps(value))
    other["label"] = "unused"
    raw["profiles"][_codec.digest(other)[:16]] = other


@pytest.mark.parametrize(("message", "edit"), EDITS, ids=[f"{i}" for i in range(len(EDITS))])
def test_every_defect_is_corrupt_state_with_its_reason(
    message: str, edit: Callable[[dict[str, Any]], None]
) -> None:
    with pytest.raises(CorruptState, match=f"invalid campaign state: .*{message}"):
        decode(_edit(edit))


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"", "not valid UTF-8 JSON"),
        (b"[]", "unsupported campaign schema"),
        (b'{"schema_version":7,"schema_version":7}', "duplicate JSON object key"),
        (b"\xff", "not valid UTF-8 JSON"),
    ],
)
def test_malformed_bytes_are_corrupt_state(data: bytes, message: str) -> None:
    with pytest.raises(CorruptState, match=message):
        decode(data)


REWRITES: list[Callable[[bytes], bytes]] = [
    lambda data: json.dumps(json.loads(data), indent=1).encode(),
    lambda data: data.rstrip(b"\n"),
    lambda data: data + b"\n",
]


@pytest.mark.parametrize("rewrite", REWRITES)
def test_noncanonical_encodings_are_rejected(rewrite: Callable[[bytes], bytes]) -> None:
    with pytest.raises(CorruptState, match="not canonically encoded"):
        decode(rewrite(encode(history())))


def test_noncanonical_values_that_decode_are_rejected() -> None:
    def keyed_by_content(raw: dict[str, Any], **target_changes: Any) -> None:
        # Key the profile by its decoded (normalized) content so only the bytes disagree.
        (value,) = raw["profiles"].values()
        value["target"].update(target_changes)
        normalized = _codec.dump(_codec.load(Profile, value))
        name = _codec.digest(normalized)[:16]
        raw["profiles"] = {name: value}
        for attempt in raw["attempts"]:
            attempt["profile"] = name

    for changes in ({"max_time_limit": "6-23:59:30"}, {"work_root": "/cluster//work/./project"}):
        with pytest.raises(CorruptState, match="not canonically encoded"):
            decode(_edit(lambda raw, changes=changes: keyed_by_content(raw, **changes)))

    raw_state, _ = submit(roster(1), ["task-0"])
    raw = json.loads(encode(raw_state))
    raw["tasks"][0]["stdin"] = "MR=="  # non-zero padding bits, decodes to b"1"
    with pytest.raises(CorruptState, match="not canonically encoded"):
        decode(reencode(raw))


def test_direct_construction_checks_attempt_time_and_task_revision() -> None:
    with pytest.raises(ValueError, match="intent time must be UTC"):
        Attempt(
            allocation_id=allocation_id(1),
            task_keys=("task-0",),
            profile=profile(),
            retry=(),
            duplicate_risk=(),
            intent_revision=1,
            intent_at=datetime(2030, 1, 1),  # noqa: DTZ001
        )
    with pytest.raises(ValueError, match="invalid task revision"):
        RegisteredTask(Task("a"), -1)


# --- performance -----------------------------------------------------------------------------


def large_state(task_count: int = 1000, attempt_count: int = 3000) -> State:
    keys = tuple(f"task-{index}" for index in range(task_count))
    attempts = tuple(
        Attempt(
            allocation_id=allocation_id(index + 1),
            task_keys=(keys[index % task_count],),
            profile=profile(),
            retry=(keys[index % task_count],) if index >= task_count else (),
            duplicate_risk=(),
            intent_revision=2 * index + 1,
            intent_at=NOW + timedelta(seconds=index),
            acceptance=AcceptanceState.ACCEPTED,
            job=JobRef(index + 1),
            outcome_revision=2 * index + 2,
        )
        for index in range(attempt_count)
    )
    roster_items = tuple(RegisteredTask(task, 0) for task in tasks(task_count))
    return State(CAMPAIGN_ID, 2 * attempt_count, roster_items, 0, attempts)


def test_decoding_three_thousand_attempts_is_fast() -> None:
    state = large_state()
    data = encode(state)
    timings: list[float] = []
    for _ in range(3):
        started = time.perf_counter()
        assert decode(data) == state
        timings.append(time.perf_counter() - started)
    assert min(timings) < 1.0, f"decode took {min(timings):.3f}s"
