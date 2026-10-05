from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from enum import Enum
from pathlib import PurePosixPath
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from servatus.campaign import _codec
from servatus.campaign._codec import FLATTEN, CodecError, key


class Color(Enum):
    RED = "red"
    BLUE = "blue"


@dataclass(frozen=True, slots=True)
class Inner:
    count: int
    flag: bool = False


@dataclass(frozen=True, slots=True)
class Outer:
    name: str
    inner: Inner = field(metadata=FLATTEN)
    payload: bytes
    path: PurePosixPath
    at: datetime
    limit: timedelta
    color: Color
    items: tuple[int, ...]
    labels: Mapping[str, str]
    renamed: int = field(metadata=key("other_name"))
    maybe: Inner | None = None


AT = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)


def outer(**changes: object) -> Outer:
    values: dict[str, object] = {
        "name": "ä",
        "inner": Inner(3, True),
        "payload": b"\0\xff",
        "path": PurePosixPath("/a/b"),
        "at": AT,
        "limit": timedelta(days=1, hours=2, minutes=3, seconds=4),
        "color": Color.BLUE,
        "items": (1, 2),
        "labels": {"z": "1", "a": "2"},
        "renamed": 7,
        "maybe": Inner(1),
    }
    values.update(changes)
    return Outer(**values)  # pyright: ignore[reportArgumentType]


def dumped(**changes: object) -> dict[str, object]:
    document = dict(cast(dict[str, object], _codec.dump(outer())))
    document.update(changes)
    return document


def test_dump_flattens_renames_and_uses_canonical_scalars() -> None:
    assert _codec.dump(outer()) == {
        "name": "ä",
        "count": 3,
        "flag": True,
        "payload": "AP8=",
        "path": "/a/b",
        "at": "2030-01-02T03:04:05+00:00",
        "limit": "1-02:03:04",
        "color": "blue",
        "items": [1, 2],
        "labels": {"z": "1", "a": "2"},
        "other_name": 7,
        "maybe": {"count": 1, "flag": False},
    }


@pytest.mark.parametrize("maybe", [None, Inner(5, True)])
def test_load_round_trips_through_canonical_json(maybe: Inner | None) -> None:
    value = outer(maybe=maybe)
    data = _codec.canonical(_codec.dump(value))
    assert _codec.load(Outer, _codec.decode_json(data)) == value


def test_canonical_json_is_sorted_compact_utf8_and_digest_is_stable() -> None:
    assert _codec.canonical({"b": 1, "a": ["é", None]}) == '{"a":["é",null],"b":1}'.encode()
    assert _codec.digest({"b": 1, "a": 2}) == _codec.digest({"a": 2, "b": 1})
    with pytest.raises(ValueError, match="not valid UTF-8"):
        _codec.canonical({"a": "\udcff"})


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"extra": 1}, r"unknown keys \['extra'\]"),
        ({"count": True}, r"\$\.count: expected int"),
        ({"flag": 1}, r"\$\.flag: expected bool"),
        ({"name": 3}, r"\$\.name: expected str"),
        ({"items": [1, "2"]}, r"\$\.items\[1\]: expected int"),
        ({"items": {"0": 1}}, r"\$\.items: expected array"),
        ({"labels": []}, r"\$\.labels: expected object"),
        ({"labels": {"a": 1}}, r"\$\.labels\.a: expected str"),
        ({"maybe": []}, r"\$\.maybe: expected object"),
        ({"payload": "!!"}, r"expected base64"),
        ({"payload": 5}, r"expected string"),
        ({"color": "green"}, r"expected one of \['red', 'blue'\]"),
        ({"at": "2030-01-02T03:04:05"}, "canonical UTC datetime"),
        ({"at": "2030-01-02T03:04:05Z"}, "canonical UTC datetime"),
        ({"at": "2030-01-02T04:04:05+01:00"}, "canonical UTC datetime"),
        ({"at": "2030-01-02 03:04:05+00:00"}, "canonical UTC datetime"),
        ({"at": "yesterday"}, "ISO datetime"),
        ({"limit": "26:03:04"}, "canonical duration"),
        ({"limit": "1-2:03:04"}, "canonical duration"),
        ({"limit": "1-24:00:00"}, r"expected duration"),
        ({"limit": "01:60:00"}, r"expected duration"),
        ({"limit": "forever"}, r"expected duration"),
    ],
)
def test_load_rejects_structural_and_noncanonical_values(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(CodecError, match=message):
        _codec.load(Outer, dumped(**changes))


@pytest.mark.parametrize("missing", ["name", "count", "other_name", "maybe"])
def test_load_rejects_missing_keys_including_flattened_and_optional(missing: str) -> None:
    document = dumped()
    del document[missing]
    with pytest.raises(CodecError, match=f"missing key '{missing}'"):
        _codec.load(Outer, document)


def test_load_rejects_non_objects_and_unsupported_hints() -> None:
    with pytest.raises(CodecError, match=r"\$: expected object"):
        _codec.load(Outer, [])
    with pytest.raises(TypeError, match="unsupported union"):
        _codec.load(int | str, 1)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="unsupported tuple"):
        _codec.load(tuple[int, str], [1, "a"])  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="unsupported mapping"):
        _codec.load(dict[int, str], {})  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="unsupported type"):
        _codec.load(float, "1.0")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (datetime(2030, 1, 1), "aware UTC"),  # noqa: DTZ001
        (datetime(2030, 1, 1, tzinfo=timezone(timedelta(hours=1))), "aware UTC"),
        ({1: "a"}, "mapping keys must be strings"),
        (1.5, "cannot encode"),
        ({"a"}, "cannot encode"),
    ],
)
def test_dump_rejects_values_without_a_canonical_form(value: object, message: str) -> None:
    with pytest.raises(CodecError, match=message):
        _codec.dump(value)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"{", "not valid UTF-8 JSON"),
        (b"\xff", "not valid UTF-8 JSON"),
        (b'{"a":1,"a":1}', "duplicate JSON object key 'a'"),
        (b'{"a":NaN}', "non-standard JSON constant NaN"),
        (b"[-Infinity]", "non-standard JSON constant -Infinity"),
        (b"[" * 200_000 + b"]" * 200_000, "not valid UTF-8 JSON"),
    ],
)
def test_decode_json_is_strict(data: bytes, message: str) -> None:
    with pytest.raises(CodecError, match=message):
        _codec.decode_json(data)


def test_decode_json_keeps_ordinary_documents() -> None:
    assert _codec.decode_json('{"a":[1,"é",null,true]}'.encode()) == {"a": [1, "é", None, True]}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("00:00:01", timedelta(seconds=1)),
        ("23:59:59", timedelta(hours=23, minutes=59, seconds=59)),
        ("99:00:00", timedelta(hours=99)),
        ("0-01:00:00", timedelta(hours=1)),
        ("7-00:00:00", timedelta(days=7)),
    ],
)
def test_parse_duration_accepts_slurm_forms(text: str, expected: timedelta) -> None:
    assert _codec.parse_duration(text) == expected


@pytest.mark.parametrize("text", ["1:00", "1-24:00:00", "00:60:00", "-1:00:00", "01-00:00:00", ""])
def test_parse_duration_rejects_other_forms(text: str) -> None:
    with pytest.raises(ValueError, match="duration"):
        _codec.parse_duration(text)


@pytest.mark.parametrize("value", [timedelta(seconds=-1), timedelta(milliseconds=1)])
def test_format_duration_requires_non_negative_whole_seconds(value: timedelta) -> None:
    with pytest.raises(ValueError, match="whole number of seconds"):
        _codec.format_duration(value)


@given(st.integers(0, 10**7))
def test_format_then_parse_is_identity_and_canonical(seconds: int) -> None:
    text = _codec.format_duration(timedelta(seconds=seconds))
    assert _codec.parse_duration(text) == timedelta(seconds=seconds)
    assert _codec.load(timedelta, text) == timedelta(seconds=seconds)
    days, _, clock = text.rpartition("-")
    assert (days == "") == (seconds < 86_400)
    assert int(clock[:2]) < 24 or not days


@given(st.binary(max_size=64), st.text(max_size=16))
def test_bytes_and_text_round_trip_through_json(payload: bytes, text: str) -> None:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return
    value = outer(payload=payload, name=text)
    reloaded = _codec.load(Outer, json.loads(_codec.canonical(_codec.dump(value))))
    assert reloaded == value
