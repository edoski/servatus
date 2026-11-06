from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import pytest
from core_helpers import CAMPAIGN_ID, NOW, allocation_id, intend, reencode, roster, submit
from support.builders import profile, tasks

from servatus.campaign import _codec
from servatus.campaign._config import Profile, Task
from servatus.campaign._state import (
    SCHEMA_VERSION,
    AcceptanceState,
    Attempt,
    RegisteredTask,
    State,
    append,
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


def test_intent_then_outcome_records_chronology_and_acceptance() -> None:
    state, first = submit(roster(2), ["task-0", "task-1"])
    attempt = state.attempt(first)
    assert attempt.acceptance is AcceptanceState.ACCEPTED and attempt.job == 101
    assert (attempt.intent_revision, attempt.outcome_revision, state.revision) == (1, 2, 2)
    state, second = intend(state, ["task-1"])
    retried = state.attempt(second)
    assert retried.acceptance is AcceptanceState.UNRESOLVED and retried.job is None


def test_outcome_is_idempotent_and_conflicts_are_rejected() -> None:
    state, identity = intend(roster(1), ["task-0"])
    accepted = record_outcome(state, identity, 42)
    assert accepted.attempt(identity).job == 42 and accepted.revision == state.revision + 1
    assert record_outcome(accepted, identity, 42) is accepted
    with pytest.raises(Conflict, match="conflicting outcome"):
        record_outcome(accepted, identity, 43)
    with pytest.raises(Conflict, match="conflicting outcome"):
        record_outcome(accepted, identity, None)
    rejected = record_outcome(state, identity, None)
    assert rejected.attempt(identity).acceptance is AcceptanceState.NOT_SUBMITTED
    assert record_outcome(rejected, identity, None) is rejected
    with pytest.raises(Conflict, match="conflicting outcome"):
        record_outcome(rejected, identity, 42)
    with pytest.raises(NotFound, match="unknown allocation"):
        record_outcome(state, allocation_id(99), None)
    with pytest.raises(NotFound, match="unknown allocation"):
        state.attempt(allocation_id(99))


@pytest.mark.parametrize("job", [0, -1])
def test_outcome_rejects_a_job_number_below_one(job: int) -> None:
    state, identity = intend(roster(1), ["task-0"])
    with pytest.raises(Conflict, match="invalid job number"):
        record_outcome(state, identity, job)


def test_outcome_rejects_a_bool_job() -> None:
    state, identity = intend(roster(1), ["task-0"])
    with pytest.raises(Conflict, match="invalid job number"):
        record_outcome(state, identity, True)


def test_outcome_survives_unrelated_authoring() -> None:
    state, identity = intend(roster(1, appendable=True), ["task-0"])
    state = seal(append(state, [Task("late")]))
    resolved = record_outcome(state, identity, 7)
    assert resolved.attempt(identity).outcome_revision == 4 == resolved.revision


def _intent(state: State, **changes: Any) -> State:
    values: dict[str, Any] = {
        "allocation_id": allocation_id(len(state.attempts) + 1),
        "task_keys": ("task-0",),
        "profile": profile(),
        "at": NOW,
    }
    values.update(changes)
    return record_intent(state, **values)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"allocation_id": "NOT-HEX"}, "invalid allocation identity"),
        ({"task_keys": ()}, "invalid attempt task references"),
        ({"task_keys": ("task-0", "task-0")}, "invalid attempt task references"),
        ({"task_keys": ("foreign",)}, "invalid attempt task references"),
        ({"task_keys": ("task-1", "task-0")}, "not in roster order"),
    ],
)
def test_record_intent_rejects_invalid_attempts_as_conflicts(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(Conflict, match=message):
        _intent(roster(5), **changes)


def test_record_intent_rejects_overlap_with_unresolved_work() -> None:
    state, _ = intend(roster(2), ["task-0"])
    with pytest.raises(Conflict, match="overlaps unresolved intent"):
        _intent(state, task_keys=("task-0", "task-1"))
    accepted, _ = submit(roster(2), ["task-0"])
    assert _intent(accepted, task_keys=("task-0", "task-1")).revision == 3


# --- documents -------------------------------------------------------------------------------


def history() -> State:
    """Accepted, retried, appended, sealed, and trailing unresolved work."""
    state, _ = submit(roster(3, appendable=True), ["task-0", "task-1"])
    state = append(state, [Task("task-3", ("x",), env={"A": "1"})])
    state, _ = submit(state, ["task-0", "task-2"])
    state = seal(state)
    state, _ = submit(state, ["task-1"], accepted=False)
    state, _ = intend(state, ["task-3"])
    return state


def test_documents_round_trip_canonically() -> None:
    state = history()
    data = encode(state)
    assert data.endswith(b"\n") and data == reencode(json.loads(data))
    assert decode(data) == state
    raw = json.loads(data)
    assert raw["schema_version"] == SCHEMA_VERSION == 8
    assert raw["tasks"][3] == {
        "task": {"key": "task-3", "args": ["x"], "stdin": "", "env": {"A": "1"}},
        "revision": 3,
    }
    assert [attempt["job"] for attempt in raw["attempts"]] == [101, 102, None, None]
    assert [attempt["outcome_revision"] for attempt in raw["attempts"]] == [2, 5, 8, None]
    assert {key for attempt in raw["attempts"] for key in attempt} == {
        "allocation_id",
        "task_keys",
        "profile",
        "intent_revision",
        "intent_at",
        "job",
        "outcome_revision",
    }
    assert [attempt.acceptance for attempt in state.attempts] == [
        AcceptanceState.ACCEPTED,
        AcceptanceState.ACCEPTED,
        AcceptanceState.NOT_SUBMITTED,
        AcceptanceState.UNRESOLVED,
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


def _unresolve_first(raw: dict[str, Any]) -> None:
    raw["attempts"][0].update(job=None, outcome_revision=None)


def _task(raw: dict[str, Any], index: int = 0) -> dict[str, Any]:
    return raw["tasks"][index]["task"]


EDITS: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
    ("invalid campaign identity", lambda r: r.update(campaign_id="C" * 32)),
    ("invalid revision", lambda r: r.update(revision=-1)),
    ("invalid seal revision", lambda r: r.update(sealed_revision=99)),
    ("invalid task revision", lambda r: r["tasks"][0].update(revision=-1)),
    ("task appended after seal", lambda r: r.update(sealed_revision=2)),
    ("duplicate task key", lambda r: _task(r, 2).update(key="task-0")),
    ("conflicting seal revision", lambda r: r.update(sealed_revision=3)),
    ("invalid event revision", lambda r: r["attempts"][0].update(outcome_revision=3)),
    (
        "duplicate allocation identity",
        lambda r: r["attempts"][1].update(allocation_id=r["attempts"][0]["allocation_id"]),
    ),
    ("invalid intent revision", lambda r: r["attempts"][1].update(intent_revision=1)),
    (
        "invalid attempt task references",
        lambda r: r["attempts"][0].update(task_keys=["task-0", "task-3"]),
    ),
    (
        "attempt tasks are not in roster order",
        lambda r: r["attempts"][1].update(task_keys=["task-2", "task-0"]),
    ),
    ("attempt overlaps unresolved intent", _unresolve_first),
    ("revision does not match recorded mutations", lambda r: r.update(revision=99)),
    ("invalid allocation identity", lambda r: r["attempts"][0].update(allocation_id="x" * 24)),
    ("unresolved intent has a job", lambda r: r["attempts"][3].update(job=1)),
    ("invalid outcome revision", lambda r: r["attempts"][0].update(outcome_revision=1)),
    ("invalid job number", lambda r: r["attempts"][0].update(job=0)),
    ("unsupported campaign schema; expected 8", lambda r: r.update(schema_version=7)),
    ("profile key does not match its content", lambda r: _rename_profile(r)),
    ("profiles do not match attempt references", lambda r: _add_unused_profile(r)),
    ("expected int", lambda r: r["attempts"][0].update(job=True)),
    ("unknown keys", lambda r: r["attempts"][0].update(acceptance="ACCEPTED")),
    ("unknown keys", lambda r: r["tasks"][0].update(key="task-0")),
    ("missing key 'env'", lambda r: _task(r).pop("env")),
    ("expected base64", lambda r: _task(r).update(stdin="!")),
    ("Task.env names must be identifiers", lambda r: _task(r).update(env={"1A": "x"})),
    ("cannot start with SERVATUS_", lambda r: _task(r).update(env={"SERVATUS_X": "1"})),
    ("Task.key cannot contain NUL", lambda r: _task(r).update(key="a\0")),
    ("canonical UTC datetime", lambda r: r["attempts"][0].update(intent_at="2030-01-01T00:00:00")),
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


def test_a_schema_7_document_is_corrupt_state() -> None:
    raw = json.loads(encode(history()))
    raw["schema_version"] = 7
    for attempt in raw["attempts"]:
        attempt["duplicate_risk"] = []
        if attempt["job"] is not None:
            attempt["job"] = {"job_id": attempt["job"], "cluster": None}
    with pytest.raises(CorruptState, match="unsupported campaign schema; expected 8"):
        decode(reencode(raw))


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
    raw["tasks"][0]["task"]["stdin"] = "MR=="  # non-zero padding bits, decodes to b"1"
    with pytest.raises(CorruptState, match="not canonically encoded"):
        decode(reencode(raw))


def test_direct_construction_checks_attempt_time_and_job() -> None:
    with pytest.raises(ValueError, match="intent time must be UTC"):
        Attempt(
            allocation_id=allocation_id(1),
            task_keys=("task-0",),
            profile=profile(),
            intent_revision=1,
            intent_at=datetime(2030, 1, 1),  # noqa: DTZ001
        )
    with pytest.raises(ValueError, match="invalid job number"):
        Attempt(
            allocation_id=allocation_id(1),
            task_keys=("task-0",),
            profile=profile(),
            intent_revision=1,
            intent_at=NOW,
            job=0,
            outcome_revision=2,
        )


# --- performance -----------------------------------------------------------------------------


def large_state(task_count: int = 1000, attempt_count: int = 3000) -> State:
    keys = tuple(f"task-{index}" for index in range(task_count))
    attempts = tuple(
        Attempt(
            allocation_id=allocation_id(index + 1),
            task_keys=(keys[index % task_count],),
            profile=profile(),
            intent_revision=2 * index + 1,
            intent_at=NOW + timedelta(seconds=index),
            job=index + 1,
            outcome_revision=2 * index + 2,
        )
        for index in range(attempt_count)
    )
    roster_items = tuple(RegisteredTask(task, 0) for task in tasks(task_count))
    return State(CAMPAIGN_ID, 2 * attempt_count, roster_items, 0, attempts)


def best_decode_seconds(state: State) -> float:
    data = encode(state)
    timings: list[float] = []
    for _ in range(3):
        started = time.perf_counter()
        assert decode(data) == state
        timings.append(time.perf_counter() - started)
    return min(timings)


def test_decoding_scales_linearly_with_history() -> None:
    # Quadrupling the history should about quadruple decode time (quadratic: about 16x).
    # A ratio, unlike a wall-clock budget, holds on slow or coverage-instrumented runners.
    small = best_decode_seconds(large_state(task_count=1000, attempt_count=1000))
    large = best_decode_seconds(large_state(task_count=1000, attempt_count=4000))
    assert large / small < 8.0, f"decode took {small:.3f}s at 1000 and {large:.3f}s at 4000"
