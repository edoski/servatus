# pyright: standard
"""Mutation equivalence: ``decode`` accepts exactly the documents an independent reference accepts.

Valid histories come from the real transitions. Each document is mutated (numbers nudged, list
items swapped/dropped/duplicated, keys removed or added, values transplanted between fields of the
same name) and re-serialized canonically. ``reference`` restates the schema-10 rules directly and
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

from core_helpers import CAMPAIGN_ID, intend, reencode, unresolved_keys
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from support.builders import profile, resources, target, tasks

from servatus.campaign import _codec
from servatus.campaign._config import Profile, Task
from servatus.campaign._state import (
    Link,
    State,
    append,
    create,
    decode,
    encode,
    record_outcome,
    seal,
)
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
ROSTER_ENTRY = {"task", "revision"}
TASK = {"key", "args", "stdin", "env"}
LINK = {"chain", "index", "count", "budget"}
ATTEMPT = {
    "allocation_id",
    "link",
    "task_keys",
    "profile",
    "intent_revision",
    "intent_at",
    "job",
    "outcome_revision",
    "slots",
    "within",
}
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
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


def link_budget(value: object) -> bool:
    """A canonical positive duration of whole minutes."""
    if not isinstance(value, str):
        return False
    match = re.fullmatch(r"(?:(0|[1-9][0-9]*)-)?([0-9]{2,}):([0-5][0-9]):00", value)
    if match is None or (match.group(1) and int(match.group(2)) > 23):
        return False
    if not match.group(1) and (len(match.group(2)) > 2 and match.group(2).startswith("0")):
        return False
    hours = int(match.group(2))
    canonical = f"{hours:02d}:{match.group(3)}:00"
    if match.group(1):
        canonical = f"{match.group(1)}-{canonical}"
    elif hours > 23:
        return False
    return value == canonical and value not in ("00:00:00", "0-00:00:00")


def reference(doc: Any) -> bool:
    try:
        check(doc)
    except Reject:
        return False
    return True


def check(doc: Any) -> None:
    need(is_dict(doc) and set(doc) == TOP)
    need(is_int(doc["schema_version"]) and doc["schema_version"] == 10)
    need(is_hex(doc["campaign_id"], 32))
    revision, sealed = doc["revision"], doc["sealed_revision"]
    need(is_int(revision) and revision >= 0)
    need(sealed is None or (is_int(sealed) and 0 <= sealed <= revision))
    need(is_list(doc["tasks"]))
    introduced: dict[str, int] = {}
    order: list[str] = []
    for item in doc["tasks"]:
        need(is_dict(item) and set(item) == ROSTER_ENTRY)
        task = item["task"]
        need(is_dict(task) and set(task) == TASK)
        need(is_text(task["key"]) and task["key"] != "")
        need(is_strings(task["args"]) and all(is_text(arg) for arg in task["args"]))
        need(canonical_base64(task["stdin"]))
        env = task["env"]
        need(is_dict(env))
        for name, value in env.items():
            need(ENV_NAME.fullmatch(name) is not None and not name.startswith("SERVATUS_"))
            need(is_text(value))
        when = item["revision"]
        need(is_int(when) and 0 <= when <= revision)
        need(not order or when >= introduced[order[-1]])
        need(sealed is None or when <= sealed)
        need(task["key"] not in introduced)
        introduced[task["key"]] = when
        order.append(task["key"])
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
        link = raw["link"]
        need(is_dict(link) and set(link) == LINK and is_hex(link["chain"], 24))
        index, count = link["index"], link["count"]
        need(is_int(index) and is_int(count) and 1 <= index <= count <= 32)
        need(link["budget"] is None or link_budget(link["budget"]))
        need(index > 1 or link["chain"] == raw["allocation_id"])
        need(
            not any(
                other["link"]["chain"] == link["chain"] and other["link"]["index"] == index
                for other in prior
            )
        )
        intent = raw["intent_revision"]
        need(is_int(intent) and 1 <= intent <= revision and intent not in events)
        need(not prior or intent > prior[-1]["intent_revision"])
        events.add(intent)
        keys, slots, within = raw["task_keys"], raw["slots"], raw["within"]
        need(is_strings(keys))
        need(len(set(keys)) == len(keys))
        need(slots is None or (is_int(slots) and slots >= 1))
        need(within is None or is_hex(within, 24))
        if within is not None:
            need(len(keys) == 1 and slots is None and count == 1)
            need(
                any(
                    other["allocation_id"] == within
                    and other["slots"] is not None
                    and other["within"] is None
                    and other["job"] is not None
                    and other["outcome_revision"] < intent
                    and other["profile"] == raw["profile"]
                    and raw["job"] in (None, other["job"])
                    for other in prior
                )
            )
        elif slots is not None:
            need(keys == [])
        else:
            need(bool(keys))
        need(all(key in introduced and introduced[key] < intent for key in keys))
        need([key for key in order if key in keys] == keys)
        need(
            not any(
                set(other["task_keys"]) & set(keys)
                for other in prior
                if other["outcome_revision"] is None or other["outcome_revision"] >= intent
            )
        )
        if index > 1:
            need(
                any(
                    other["link"]["chain"] == link["chain"]
                    and other["link"]["index"] == index - 1
                    and other["link"]["count"] == count
                    and other["link"]["budget"] == link["budget"]
                    and other["task_keys"] == keys
                    and other["slots"] == slots
                    and other["job"] is not None
                    and other["outcome_revision"] < intent
                    for other in prior
                )
            )
        need(canonical_utc(raw["intent_at"]))
        outcome, job = raw["outcome_revision"], raw["job"]
        need(job is None or (is_int(job) and job >= 1))
        if outcome is None:
            need(job is None)
        else:
            need(is_int(outcome) and intent < outcome <= revision and outcome not in events)
            events.add(outcome)
        need(isinstance(raw["profile"], str) and raw["profile"] in loaded)
        used.add(raw["profile"])
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
        operation = draw(
            st.sampled_from(
                ["append", "seal", "intent", "intent", "outcome", "next link", "lease", "step"]
            )
        )
        unresolved = [a for a in state.attempts if a.outcome_revision is None]
        if operation == "append" and not state.sealed:
            env = draw(st.sampled_from([{}, {"A": "1"}, {"Z_9": "é"}]))
            state = append(state, [Task(f"extra-{extra}", ("run",), stdin=b"\x00x", env=env)])
            extra += 1
        elif operation == "seal":
            state = seal(state)
        elif operation == "intent":
            free = [task.key for task in state.tasks if task.key not in unresolved_keys(state)]
            if not free:
                continue
            chosen_profile = draw(st.sampled_from(PROFILES))
            limit = chosen_profile.target.max_tasks_per_allocation
            chosen = draw(st.lists(st.sampled_from(free), min_size=1, max_size=limit, unique=True))
            keys = [key for key in free if key in chosen]
            identity = f"{len(state.attempts) + 1:024x}"
            count = draw(st.sampled_from([1, 1, 3]))
            budget = draw(st.sampled_from([None, timedelta(hours=30)]))
            link = Link(identity, 1, count, budget)
            state, _ = intend(state, keys, profile_value=chosen_profile, link=link)
        elif operation == "lease":
            identity = f"{len(state.attempts) + 1:024x}"
            link = Link(identity, 1, draw(st.sampled_from([1, 2])))
            slots = draw(st.integers(1, 3))
            state, _ = intend(state, (), profile_value=PROFILES[0], link=link, slots=slots)
        elif operation == "step":
            free = [task.key for task in state.tasks if task.key not in unresolved_keys(state)]
            leases = [a for a in state.attempts if a.slots is not None and a.within is None]
            leases = [a for a in leases if a.job is not None]
            if not free or not leases:
                continue
            lease = draw(st.sampled_from(leases))
            state, _ = intend(
                state,
                (draw(st.sampled_from(free)),),
                profile_value=lease.profile,
                within=lease.allocation_id,
            )
        elif operation == "next link":
            busy = unresolved_keys(state)
            taken = {(a.link.chain, a.link.index) for a in state.attempts}
            ready = [
                a
                for a in state.attempts
                if a.job is not None
                and a.link.index < a.link.count
                and (a.link.chain, a.link.index + 1) not in taken
                and not busy & set(a.task_keys)
            ]
            if ready:
                previous = draw(st.sampled_from(ready))
                old = previous.link
                link = Link(old.chain, old.index + 1, old.count, old.budget)
                state, _ = intend(
                    state,
                    previous.task_keys,
                    profile_value=previous.profile,
                    link=link,
                    slots=previous.slots,
                )
        elif operation == "outcome" and unresolved:
            attempt = draw(st.sampled_from(unresolved))
            job = draw(st.sampled_from([None, 7, 8]))
            if attempt.within is not None and job is not None:
                job = state.attempt(attempt.within).job
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
    "30:00:00",
    "1-06:00:00",
    3,
    33,
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


def clone(value: Any) -> Any:
    return json.loads(json.dumps(value))


def walk(doc: Any, path: tuple[Any, ...]) -> Any:
    for step in path:
        doc = doc[step]
    return doc


REVISIONS = {"revision", "sealed_revision", "intent_revision", "outcome_revision"}


def open_gap(doc: Any, rng: random.Random) -> None:
    """Shift every revision at or above a random point up by one, leaving a free revision."""
    if not is_int(doc.get("revision")) or doc["revision"] < 0:
        return
    start = rng.randint(1, doc["revision"] + 1)
    for path in paths(doc):
        if path and path[-1] in REVISIONS:
            parent = walk(doc, path[:-1])
            if is_int(parent[path[-1]]) and parent[path[-1]] >= start:
                parent[path[-1]] += 1


def mutate(base: Any, rng: random.Random) -> Any:
    doc = clone(base)
    for _ in range(rng.choice((1, 1, 2, 3))):
        every = [path for path in paths(doc) if path]
        path = rng.choice(every)
        parent, leaf = walk(doc, path[:-1]), path[-1]
        value = parent[leaf]
        operation = rng.randrange(10)
        if operation == 8:
            open_gap(doc, rng)
        elif operation == 9 and is_int(value) and value >= 0 and leaf in REVISIONS:
            parent[leaf] = rng.randint(0, value + 1)
        elif operation == 0 and is_int(value):
            parent[leaf] = value + rng.choice((-2, -1, 1, 2))
        elif operation == 1 and isinstance(parent, list) and len(parent) > 1:
            other = rng.randrange(len(parent))
            parent[leaf], parent[other] = parent[other], parent[leaf]
        elif operation == 2 and isinstance(parent, list):
            del parent[leaf]
        elif operation == 3 and isinstance(parent, list):
            parent.insert(leaf, clone(value))
        elif operation == 4 and isinstance(parent, dict):
            del parent[leaf]
        elif operation == 5 and isinstance(parent, dict):
            parent["extra"] = 1
        else:
            same = [walk(doc, other) for other in every if other[-1] == leaf and other != path]
            # A shared SCALARS list or dict edited in place would change later examples.
            pool = same if same and rng.random() < 0.6 else SCALARS
            parent[leaf] = clone(rng.choice(pool))
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
    state = record_outcome(state, first, 41)
    state = append(state, [Task("task-4", env={"A": "1"})])
    state, second = intend(state, ["task-0", "task-2", "task-4"])
    state = record_outcome(state, second, 42)
    state, third = intend(state, ["task-3"])
    state = record_outcome(seal(state), third, None)
    state, _ = intend(state, ["task-1", "task-3"], profile_value=PROFILES[1])
    state, chain = intend(state, ["task-4"], link=Link(f"{5:024x}", 1, 3, timedelta(hours=2)))
    state = record_outcome(state, chain, 43)
    state, _ = intend(state, ["task-4"], link=Link(chain, 2, 3, timedelta(hours=2)))
    state, lease = intend(state, (), link=Link(f"{7:024x}", 1, 2), slots=2)
    state = record_outcome(state, lease, 44)
    state, step = intend(state, ["task-2"], within=lease)
    state = record_outcome(state, step, 44)
    state, _ = intend(state, (), link=Link(lease, 2, 2), slots=2)
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
