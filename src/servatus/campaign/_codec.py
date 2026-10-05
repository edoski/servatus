"""Strict dataclass <-> JSON codec, canonical JSON, and Slurm durations.

Structure lives here; meaning lives in each dataclass's constructor. Decoding rejects unknown and
missing keys, booleans used as integers, and non-canonical datetimes and durations.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import hashlib
import json
import re
import types
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from enum import Enum
from functools import cache
from pathlib import PurePosixPath
from typing import Any, TypeVar, Union, cast, get_args, get_origin, get_type_hints

T = TypeVar("T")
FLATTEN: Mapping[str, object] = {"codec.flatten": True}

_DURATION = re.compile(r"(?:(0|[1-9][0-9]*)-)?([0-9]+):([0-5][0-9]):([0-5][0-9])\Z")


class CodecError(ValueError):
    """A document does not match the expected structure."""


def key(name: str) -> Mapping[str, object]:
    """Field metadata that stores an attribute under a different document key."""
    return {"codec.key": name}


# --- Durations -------------------------------------------------------------------------------


def parse_duration(text: str) -> timedelta:
    """Parse ``[days-]hours:minutes:seconds``; hours stay below 24 when days are present."""
    match = _DURATION.fullmatch(text)
    if match is None:
        raise ValueError("duration must use [days-]hours:minutes:seconds")
    days = int(match.group(1) or 0)
    hours = int(match.group(2))
    if match.group(1) is not None and hours > 23:
        raise ValueError("duration hours must be below 24 when days are present")
    return timedelta(
        days=days, hours=hours, minutes=int(match.group(3)), seconds=int(match.group(4))
    )


def format_duration(value: timedelta) -> str:
    """Format a non-negative whole-second duration canonically for Slurm."""
    total = value // timedelta(seconds=1)
    if total < 0 or timedelta(seconds=total) != value:
        raise ValueError("duration must be a non-negative whole number of seconds")
    days, remainder = divmod(total, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}-{clock}" if days else clock


# --- Canonical JSON --------------------------------------------------------------------------


def canonical(value: object) -> bytes:
    """Sorted, compact, UTF-8 JSON. Raises ValueError for strings that are not valid UTF-8."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    except UnicodeEncodeError as error:
        raise ValueError("document contains text that is not valid UTF-8") from error


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def decode_json(data: bytes) -> object:
    """Parse strict JSON, rejecting duplicate object keys, NaN/Infinity, and runaway nesting."""

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for name, item in items:
            if name in result:
                raise CodecError(f"duplicate JSON object key {name!r}")
            result[name] = item
        return result

    def constant(name: str) -> object:
        raise CodecError(f"non-standard JSON constant {name}")

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise CodecError("document is not valid UTF-8 JSON") from error


# --- Encoding --------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class _Slot:
    attr: str
    key: str
    hint: Any
    flatten: bool


@cache
def _slots(cls: type) -> tuple[_Slot, ...]:
    hints = get_type_hints(cls)
    return tuple(
        _Slot(
            field.name,
            cast(str, field.metadata.get("codec.key", field.name)),
            hints[field.name],
            bool(field.metadata.get("codec.flatten")),
        )
        for field in dataclasses.fields(cls)
    )


@cache
def _keys(cls: type) -> frozenset[str]:
    names: set[str] = set()
    for slot in _slots(cls):
        names.update(_keys(slot.hint) if slot.flatten else {slot.key})
    return frozenset(names)


def dump(value: object) -> object:
    """Encode a dataclass (or supported value) as JSON-compatible data."""
    return _encoder(type(value))(value)


def _same(value: object) -> object:
    return value


def _datetime(value: datetime) -> str:
    if value.utcoffset() != timedelta(0):
        raise CodecError("datetimes must be aware UTC")
    return value.isoformat()


def _mapping(value: Mapping[object, object]) -> dict[str, object]:
    if any(type(name) is not str for name in value):
        raise CodecError("mapping keys must be strings")
    return {cast(str, name): dump(item) for name, item in value.items()}


@cache
def _encoder(tp: type[Any]) -> Callable[[Any], object]:
    if issubclass(tp, Enum):
        return lambda value: value.value
    if tp in (str, int, bool, type(None)):
        return _same
    if tp is bytes:
        return lambda value: base64.b64encode(value).decode("ascii")
    if issubclass(tp, PurePosixPath):
        return str
    if issubclass(tp, datetime):
        return _datetime
    if issubclass(tp, timedelta):
        return format_duration
    if dataclasses.is_dataclass(tp):
        plan = tuple((slot.attr, slot.key, slot.flatten) for slot in _slots(tp))

        def encode(value: object) -> dict[str, object]:
            out: dict[str, object] = {}
            for attr, name, flatten in plan:
                item: object = getattr(value, attr)
                if flatten:
                    out.update(cast(dict[str, object], dump(item)))
                else:
                    out[name] = dump(item)
            return out

        return encode
    if issubclass(tp, (tuple, list)):
        return lambda value: [dump(item) for item in cast(Sequence[object], value)]
    if issubclass(tp, Mapping):
        return _mapping
    raise CodecError(f"cannot encode {tp!r}")


# --- Decoding --------------------------------------------------------------------------------


def load(cls: type[T], data: object) -> T:
    """Decode ``data`` as ``cls``. Constructors perform semantic validation."""
    return cast(T, _load(cls, data, "$"))


def _fail(path: str, expected: str) -> CodecError:
    return CodecError(f"{path}: expected {expected}")


def _load(hint: Any, data: object, path: str) -> object:
    if dataclasses.is_dataclass(hint):
        return _object(cast(type, hint), data, path)
    origin, args = get_origin(hint), get_args(hint)
    if origin in (Union, types.UnionType):
        inner = [arg for arg in args if arg is not type(None)]
        if len(args) != 2 or len(inner) != 1:
            raise TypeError(f"unsupported union {hint!r}")
        return None if data is None else _load(inner[0], data, path)
    if origin in (tuple, Sequence):
        if origin is tuple and (len(args) != 2 or args[1] is not Ellipsis):
            raise TypeError(f"unsupported tuple {hint!r}")
        if type(data) is not list:
            raise _fail(path, "array")
        items = cast(list[object], data)
        return tuple(_load(args[0], item, f"{path}[{index}]") for index, item in enumerate(items))
    if origin in (dict, Mapping):
        if args[0] is not str:
            raise TypeError(f"unsupported mapping {hint!r}")
        if type(data) is not dict:
            raise _fail(path, "object")
        pairs = cast(dict[str, object], data)
        return {name: _load(args[1], item, f"{path}.{name}") for name, item in pairs.items()}
    if hint in (bool, int, str):
        if type(data) is not hint:
            raise _fail(path, hint.__name__)
        return data
    if type(data) is not str:
        raise _fail(path, "string")
    text = data
    if hint is bytes:
        try:
            return base64.b64decode(text, validate=True)
        except (ValueError, binascii.Error) as error:
            raise _fail(path, "base64") from error
    if hint is PurePosixPath:
        return PurePosixPath(text)
    if hint is datetime:
        try:
            at = datetime.fromisoformat(text)
        except ValueError as error:
            raise _fail(path, "ISO datetime") from error
        if at.utcoffset() != timedelta(0) or at.isoformat() != text:
            raise _fail(path, "canonical UTC datetime")
        return at
    if hint is timedelta:
        try:
            duration = parse_duration(text)
        except ValueError as error:
            raise _fail(path, "duration") from error
        if format_duration(duration) != text:
            raise _fail(path, "canonical duration")
        return duration
    if isinstance(hint, type) and issubclass(hint, Enum):
        try:
            return hint(text)
        except ValueError as error:
            raise _fail(path, f"one of {[member.value for member in hint]}") from error
    raise TypeError(f"unsupported type {hint!r}")


def _object(cls: type, data: object, path: str) -> object:
    if type(data) is not dict:
        raise _fail(path, "object")
    mapping = cast(dict[str, object], data)
    if unknown := mapping.keys() - _keys(cls):
        raise CodecError(f"{path}: unknown keys {sorted(unknown)}")
    values: dict[str, object] = {}
    for slot in _slots(cls):
        if slot.flatten:
            inner = {name: mapping[name] for name in _keys(slot.hint) if name in mapping}
            values[slot.attr] = _object(slot.hint, inner, path)
        elif slot.key in mapping:
            values[slot.attr] = _load(slot.hint, mapping[slot.key], f"{path}.{slot.key}")
        else:
            raise CodecError(f"{path}: missing key {slot.key!r}")
    return cls(**values)
