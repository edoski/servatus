# pyright: standard
"""Mutation equivalence: ``decode`` accepts exactly the documents an independent reference accepts.

Valid histories come from the real transitions. Each document is mutated (numbers nudged, list
items swapped/dropped/duplicated, keys removed or added, values transplanted between fields of the
same name) and re-serialized canonically. ``reference`` restates the schema-7 rules directly and
quadratically, in the style of the 0.11 decoder; only Profile decoding reuses the codec.
"""

from __future__ import annotations

import base64
import binascii
import json
import random
import re
from datetime import datetime, timedelta
from typing import Any

from core_helpers import CAMPAIGN_ID, intend, reencode
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from support.builders import profile, resources, target, tasks

from servatus.campaign import _codec
from servatus.campaign._config import Profile, Task
from servatus.campaign._evidence import JobRef
from servatus.campaign._state import State, append, create, decode, encode, record_outcome, seal
from servatus.errors import ConfigurationError, CorruptState

TOP = {
    "schema_version",
    "campaign_id",
    "revision",
    "sealed_revision",
    "tasks",
    "profiles",
    "attempts",
}
TASK = {"key", "args", "stdin", "env", "revision"}
ATTEMPT = {
    "allocation_id",
    "task_keys",
    "profile",
    "retry",
    "duplicate_risk",
    "intent_revision",
    "intent_at",
    "acceptance",
    "job",
    "outcome_revision",
}
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
PROFILES = (
    profile(),
    profile(
        target(
            container=None,
            host=None,
            gpu_gres=None,
            max_gpus_per_allocation=0,
            max_tasks_per_allocation=2,
        ),
        resources(cpus=1, memory_mib=10, gpus=0, time_limit="00:30:00"),
        label="cpu/ü",
    ),
)


class Reject(Exception):
    pass


def need(condition: bool) -> None:
    if not condition:
        raise Reject


def is_int(value: object) -> bool:
    return type(value) is int


def is_dict(value: object) -> bool:
    return type(value) is dict


def is_list(value: object) -> bool:
    return type(value) is list


def is_text(value: object) -> bool:
    if not isinstance(value, str) or "\0" in value:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def is_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


def is_strings(value: Any) -> bool:
    return is_list(value) and all(isinstance(item, str) for item in value)


def canonical_base64(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return base64.b64encode(base64.b64decode(value, validate=True)).decode() == value
    except (ValueError, binascii.Error):
        return False


def canonical_utc(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        at = datetime.fromisoformat(value)
    except ValueError:
        return False
    return at.utcoffset() == timedelta(0) and at.isoformat() == value


def reference(doc: Any) -> bool:
    try:
        check(doc)
    except Reject:
        return False
    return True


def check(doc: Any) -> None:
    need(is_dict(doc) and set(doc) == TOP)
    need(is_int(doc["schema_version"]) and doc["schema_version"] == 7)
    need(is_hex(doc["campaign_id"], 32))
    revision, sealed = doc["revision"], doc["sealed_revision"]
    need(is_int(revision) and revision >= 0)
    need(sealed is None or (is_int(sealed) and 0 <= sealed <= revision))
    need(is_list(doc["tasks"]))
    introduced: dict[str, int] = {}
    order: list[str] = []
    for item in doc["tasks"]:
        need(is_dict(item) and set(item) == TASK)
        need(is_text(item["key"]) and item["key"] != "")
        need(is_strings(item["args"]) and all(is_text(arg) for arg in item["args"]))
        need(canonical_base64(item["stdin"]))
        env = item["env"]
        need(is_dict(env))
        for name, value in env.items():
            need(ENV_NAME.fullmatch(name) is not None and not name.startswith("SERVATUS_"))
            need(is_text(value))
        when = item["revision"]
        need(is_int(when) and 0 <= when <= revision)
        need(not order or when >= introduced[order[-1]])
        need(sealed is None or when <= sealed)
        need(item["key"] not in introduced)
        introduced[item["key"]] = when
        order.append(item["key"])
    events = {when for when in introduced.values() if when}
    if sealed:
        need(sealed not in events)
        events.add(sealed)
    need(is_dict(doc["profiles"]))
    loaded: dict[str, Profile] = {}
    for name, value in doc["profiles"].items():
        try:
            decoded = _codec.load(Profile, value)
        except (ValueError, TypeError, ConfigurationError):
            raise Reject from None
        need(_codec.dump(decoded) == value and _codec.digest(value)[:16] == name)
        loaded[name] = decoded
    need(is_list(doc["attempts"]))
    prior: list[dict[str, Any]] = []
    used: set[str] = set()
    for raw in doc["attempts"]:
        need(is_dict(raw) and set(raw) == ATTEMPT)
        need(is_hex(raw["allocation_id"], 24))
        need(all(raw["allocation_id"] != other["allocation_id"] for other in prior))
        intent = raw["intent_revision"]
        need(is_int(intent) and 1 <= intent <= revision and intent not in events)
        need(not prior or intent > prior[-1]["intent_revision"])
        events.add(intent)
        keys, retry, ack = raw["task_keys"], raw["retry"], raw["duplicate_risk"]
        need(is_strings(keys) and is_strings(retry) and is_strings(ack))
        need(bool(keys) and len(set(keys)) == len(keys))
        need(all(key in introduced and introduced[key] < intent for key in keys))
        need([key for key in order if key in keys] == keys)
        need([key for key in keys if key in retry] == retry)
        need([key for key in retry if key in ack] == ack)
        earlier = {
            key
            for other in prior
            if other["acceptance"] == "ACCEPTED" and other["outcome_revision"] < intent
            for key in other["task_keys"]
        }
        need([key for key in keys if key in earlier] == retry)
        need(
            not any(
                set(other["task_keys"]) & set(keys)
                for other in prior
                if other["outcome_revision"] is None or other["outcome_revision"] >= intent
            )
        )
        need(canonical_utc(raw["intent_at"]))
        acceptance, outcome, job = raw["acceptance"], raw["outcome_revision"], raw["job"]
        need(acceptance in ("UNRESOLVED", "ACCEPTED", "NOT_SUBMITTED"))
        if acceptance == "UNRESOLVED":
            need(outcome is None and job is None)
        else:
            need(is_int(outcome) and intent < outcome <= revision and outcome not in events)
            events.add(outcome)
            if acceptance == "ACCEPTED":
                need(is_dict(job) and set(job) == {"job_id", "cluster"})
                need(is_int(job["job_id"]) and job["job_id"] >= 1)
                cluster = job["cluster"]
                need(
                    cluster is None or (isinstance(cluster, str) and bool(TOKEN.fullmatch(cluster)))
                )
            else:
                need(job is None)
        need(isinstance(raw["profile"], str) and raw["profile"] in loaded)
        used.add(raw["profile"])
        chosen = loaded[raw["profile"]]
        count, request, route = len(keys), chosen.resources, chosen.target
        need(count <= route.max_tasks_per_allocation)
        need(count * request.cpus <= route.max_cpus_per_allocation)
        need(count * request.memory_mib <= route.max_memory_mib_per_allocation)
        need(count * request.gpus <= route.max_gpus_per_allocation)
        need(request.time_limit <= route.max_time_limit)
        prior.append(raw)
    need(used == set(doc["profiles"]))
    need(revision == max(events, default=0))


# --- valid histories -------------------------------------------------------------------------


@st.composite
def histories(draw: st.DrawFn) -> State:
    count = draw(st.integers(1, 4))
    state = create(CAMPAIGN_ID, tasks(count), appendable=draw(st.booleans()))
    extra = 0
    for _ in range(draw(st.integers(0, 10))):
        operation = draw(st.sampled_from(["append", "seal", "intent", "intent", "outcome"]))
        unresolved = [a for a in state.attempts if a.acceptance.value == "UNRESOLVED"]
        if operation == "append" and not state.sealed:
            env = draw(st.sampled_from([{}, {"A": "1"}, {"Z_9": "é"}]))
            state = append(state, [Task(f"extra-{extra}", ("run",), stdin=b"\x00x", env=env)])
            extra += 1
        elif operation == "seal":
            state = seal(state)
        elif operation == "intent":
            busy = {key for attempt in unresolved for key in attempt.task_keys}
            free = [task.key for task in state.tasks if task.key not in busy]
            if not free:
                continue
            chosen_profile = draw(st.sampled_from(PROFILES))
            limit = chosen_profile.target.max_tasks_per_allocation
            chosen = draw(st.lists(st.sampled_from(free), min_size=1, max_size=limit, unique=True))
            keys = [key for key in free if key in chosen]
            state, _ = intend(
                state,
                keys,
                ack=draw(st.sets(st.sampled_from(keys))),
                profile_value=chosen_profile,
            )
        elif operation == "outcome" and unresolved:
            attempt = draw(st.sampled_from(unresolved))
            job = draw(st.sampled_from([None, JobRef(7), JobRef(8, "alpha")]))
            state = record_outcome(state, attempt.allocation_id, job)
    return state


# --- mutations -------------------------------------------------------------------------------

SCALARS: list[object] = [
    None,
    True,
    False,
    0,
    1,
    2,
    -1,
    7,
    1.0,
    "",
    "x",
    "\0",
    [],
    {},
    "ACCEPTED",
    "NOT_SUBMITTED",
    "UNRESOLVED",
    "0" * 24,
    "f" * 24,
    "AA==",
    "MR==",
    "2030-01-01T00:00:00+00:00",
    "2030-01-01T00:00:00",
    "/a//b",
    "01:00:30",
    {"job_id": 1, "cluster": None},
]


def paths(node: Any, prefix: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    found = [prefix]
    if isinstance(node, dict):
        for name, value in node.items():
            found += paths(value, (*prefix, name))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found += paths(value, (*prefix, index))
    return found


def walk(doc: Any, path: tuple[Any, ...]) -> Any:
    for step in path:
        doc = doc[step]
    return doc


REVISIONS = {"revision", "sealed_revision", "intent_revision", "outcome_revision"}


def open_gap(doc: Any, rng: random.Random) -> None:
    """Shift every revision at or above a random point up by one, leaving a free revision."""
    if not is_int(doc.get("revision")):
        return
    start = rng.randint(1, doc["revision"] + 1)
    for path in paths(doc):
        if path and path[-1] in REVISIONS:
            parent = walk(doc, path[:-1])
            if is_int(parent[path[-1]]) and parent[path[-1]] >= start:
                parent[path[-1]] += 1


def mutate(base: Any, rng: random.Random) -> Any:
    doc = json.loads(json.dumps(base))
    for _ in range(rng.choice((1, 1, 2, 3))):
        every = [path for path in paths(doc) if path]
        path = rng.choice(every)
        parent, leaf = walk(doc, path[:-1]), path[-1]
        value = parent[leaf]
        operation = rng.randrange(10)
        if operation == 8:
            open_gap(doc, rng)
        elif operation == 9 and is_int(value) and leaf in REVISIONS:
            parent[leaf] = rng.randint(0, value + 1)
        elif operation == 0 and is_int(value):
            parent[leaf] = value + rng.choice((-2, -1, 1, 2))
        elif operation == 1 and isinstance(parent, list) and len(parent) > 1:
            other = rng.randrange(len(parent))
            parent[leaf], parent[other] = parent[other], parent[leaf]
        elif operation == 2 and isinstance(parent, list):
            del parent[leaf]
        elif operation == 3 and isinstance(parent, list):
            parent.insert(leaf, json.loads(json.dumps(value)))
        elif operation == 4 and isinstance(parent, dict):
            del parent[leaf]
        elif operation == 5 and isinstance(parent, dict):
            parent["extra"] = 1
        else:
            same = [
                json.loads(json.dumps(walk(doc, other)))
                for other in every
                if other[-1] == leaf and other != path
            ]
            parent[leaf] = rng.choice(same) if same and rng.random() < 0.6 else rng.choice(SCALARS)
    return doc


def classify(doc: Any) -> tuple[bool, bool] | None:
    """(reference verdict, decode verdict), or None when the mutation is not serializable."""
    try:
        data = reencode(doc)
    except ValueError:
        return None
    try:
        decoded = decode(data)
    except CorruptState:
        return reference(doc), False
    assert encode(decoded) == data
    return reference(doc), True


@settings(
    max_examples=250,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(histories(), st.randoms(use_true_random=False))
def test_decode_rejects_exactly_the_invalid_mutations(state: State, rng: random.Random) -> None:
    base = json.loads(encode(state))
    assert reference(base)
    assert decode(encode(state)) == state
    for _ in range(12):
        doc = mutate(base, rng)
        verdicts = classify(doc)
        if verdicts is not None:
            expected, actual = verdicts
            assert expected == actual, json.dumps(doc)


def test_mutation_corpus_exercises_both_verdicts() -> None:
    rng = random.Random(7)
    state = create(CAMPAIGN_ID, tasks(4), appendable=True)
    state, first = intend(state, ["task-0", "task-1"], profile_value=PROFILES[1])
    state = record_outcome(state, first, JobRef(41, "alpha"))
    state = append(state, [Task("task-4", env={"A": "1"})])
    state, second = intend(state, ["task-0", "task-2", "task-4"], ack=["task-0"])
    state = record_outcome(state, second, JobRef(42))
    state, third = intend(state, ["task-3"])
    state = record_outcome(seal(state), third, None)
    state, _ = intend(state, ["task-1", "task-3"], profile_value=PROFILES[1])
    base = json.loads(encode(state))
    accepted = rejected = 0
    for _ in range(4000):
        verdicts = classify(mutate(base, rng))
        if verdicts is None:
            continue
        expected, actual = verdicts
        assert expected == actual
        accepted += actual
        rejected += not actual
    assert accepted >= 100 and rejected >= 2000, (accepted, rejected)
