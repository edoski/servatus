"""Campaign inputs: Tasks, per-Task resources, Slurm targets, and named profiles."""

from __future__ import annotations

import dataclasses
import os
import re
import tomllib
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Self, cast

from ..errors import ConfigurationError, NotFound
from ._codec import parse_duration

StrPath = str | os.PathLike[str]
PosixInput = str | PurePosixPath

TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_HOST = re.compile(r"(?:[A-Za-z0-9_][A-Za-z0-9._-]*@)?[A-Za-z0-9][A-Za-z0-9._-]*\Z")
CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_GRES = re.compile(r"gpu(?::(?![0-9]+\Z)[A-Za-z0-9][A-Za-z0-9._-]*)?\Z")
RESERVED_ENV_PREFIX = "SERVATUS_"
DEFAULT_MAX_SCRIPT_BYTES = 1024 * 1024
_MINUTE = timedelta(minutes=1)


# --- Validation helpers ----------------------------------------------------------------------


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)


def _text(value: object, name: str, *, empty: bool = False) -> str:
    """A string that is valid UTF-8 and contains no NUL."""
    _require(isinstance(value, str), f"{name} must be a string")
    text = cast(str, value)
    _require(empty or bool(text), f"{name} must be nonempty")
    _require("\0" not in text, f"{name} cannot contain NUL")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise ConfigurationError(f"{name} must be valid UTF-8 text") from None
    return text


def task_keys(value: object, name: str) -> tuple[str, ...]:
    """A collection (not a single string) of Task key strings, as a tuple."""
    _require(
        isinstance(value, Collection) and not isinstance(value, (str, bytes)),
        f"{name} must be a collection of Task keys",
    )
    items = tuple(cast(Collection[object], value))
    _require(
        all(isinstance(item, str) for item in items), f"{name} must contain only Task key strings"
    )
    return cast(tuple[str, ...], items)


def _integer(value: object, name: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}")
    return value


def _token(value: object, name: str) -> str:
    if not isinstance(value, str) or TOKEN.fullmatch(value) is None:
        raise ConfigurationError(f"{name} must be one safe site token")
    return value


def _optional_token(value: object, name: str) -> str | None:
    return None if value is None else _token(value, name)


def _plain(path: PurePosixPath, name: str, forbidden: str, use: str) -> PurePosixPath:
    """``path``, unless it contains a character that ``use`` would interpret."""
    if found := sorted(set(forbidden) & set(str(path))):
        shown = " or ".join(repr(character) for character in found)
        raise ConfigurationError(f"{name} cannot contain {shown}: {use}")
    return path


def _absolute(value: object, name: str) -> PurePosixPath:
    _require(isinstance(value, (str, PurePosixPath)), f"{name} must be an absolute POSIX path")
    raw = _text(str(cast(PosixInput, value)), name)
    path = PurePosixPath(raw)
    _require(
        CONTROL.search(raw) is None and path.is_absolute() and ".." not in path.parts,
        f"{name} must be an absolute POSIX path without parent traversal or control characters",
    )
    return path


def _strings(value: object, name: str) -> tuple[str, ...]:
    _require(
        isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray, Mapping)),
        f"{name} must be a sequence of strings",
    )
    return tuple(_text(item, name, empty=True) for item in cast(Iterable[object], value))


def duration(value: object, name: str) -> timedelta:
    """A positive duration from ``[days-]hours:minutes:seconds`` text or a ``timedelta``."""
    if isinstance(value, str):
        try:
            parsed = parse_duration(value)
        except ValueError as error:
            raise ConfigurationError(f"{name}: {error}") from None
    elif isinstance(value, timedelta):
        parsed = value
    else:
        raise ConfigurationError(f"{name} must be a duration string or timedelta")
    _require(parsed > timedelta(0), f"{name} must be positive, not unlimited")
    _require(parsed % timedelta(seconds=1) == timedelta(0), f"{name} must be whole seconds")
    return parsed


def whole_minutes(value: timedelta) -> timedelta:
    """Round upward once to Slurm's whole-minute resolution."""
    return -(-value // _MINUTE) * _MINUTE


# --- Task ------------------------------------------------------------------------------------


def _restore_task(key: str, args: tuple[str, ...], stdin: bytes, env: dict[str, str]) -> Task:
    return Task(key, args, stdin=stdin, env=env)


@dataclass(frozen=True, slots=True, init=False)
class Task:
    """One opaque unit of work: a stable key, an argument vector, environment, and stdin bytes."""

    key: str
    args: tuple[str, ...]
    stdin: bytes
    env: Mapping[str, str]

    def __init__(
        self,
        key: str,
        args: Iterable[str] = (),
        *,
        stdin: bytes = b"",
        env: Mapping[str, str] | None = None,
    ) -> None:
        _text(key, "Task.key")
        frozen_args = _strings(args, "Task.args")
        _require(isinstance(stdin, bytes), "Task.stdin must be bytes")
        values: object = {} if env is None else env
        _require(isinstance(values, Mapping), "Task.env must map environment names to strings")
        frozen_env: dict[str, str] = {}
        for name, item in sorted(cast(Mapping[object, object], values).items(), key=_by_name):
            _require(
                isinstance(name, str) and _ENV_NAME.fullmatch(name) is not None,
                "Task.env names must be identifiers",
            )
            name = cast(str, name)
            _require(
                not name.startswith(RESERVED_ENV_PREFIX),
                f"Task.env names cannot start with {RESERVED_ENV_PREFIX}",
            )
            frozen_env[name] = _text(item, f"Task.env[{name}]", empty=True)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "args", frozen_args)
        object.__setattr__(self, "stdin", stdin)
        object.__setattr__(self, "env", MappingProxyType(frozen_env))

    def __hash__(self) -> int:
        return hash((self.key, self.args, self.stdin, tuple(self.env.items())))

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (_restore_task, (self.key, self.args, self.stdin, dict(self.env)))


def _by_name(item: tuple[object, object]) -> str:
    return str(item[0])


# --- Resources -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, init=False)
class Resources:
    """The homogeneous per-Task request used for one plan.

    ``time_limit`` is stored rounded upward to whole minutes, as Slurm enforces it.
    ``signal_before_end`` asks Slurm to send SIGUSR1 to each Task that long before the limit.
    """

    cpus: int
    memory_mib: int
    time_limit: timedelta
    gpus: int = 0
    signal_before_end: timedelta | None = None

    def __init__(
        self,
        *,
        cpus: int,
        memory_mib: int,
        time_limit: str | timedelta,
        gpus: int = 0,
        signal_before_end: str | timedelta | None = None,
    ) -> None:
        limit = whole_minutes(duration(time_limit, "time_limit"))
        signal = None
        if signal_before_end is not None:
            signal = duration(signal_before_end, "signal_before_end")
            _require(signal < limit, "signal_before_end must be shorter than time_limit")
            _require(signal <= timedelta(seconds=65_535), "signal_before_end exceeds Slurm's bound")
        object.__setattr__(self, "cpus", _integer(cpus, "cpus", minimum=1))
        object.__setattr__(self, "memory_mib", _integer(memory_mib, "memory_mib", minimum=1))
        object.__setattr__(self, "time_limit", limit)
        object.__setattr__(self, "gpus", _integer(gpus, "gpus", minimum=0))
        object.__setattr__(self, "signal_before_end", signal)


# --- Target ----------------------------------------------------------------------------------


def _bind(value: object) -> str:
    text = _text(value, "bind")
    parts = text.split(":")
    _require(1 <= len(parts) <= 3, "bind must be SOURCE[:DESTINATION[:ro|rw]]")
    for part in parts[:2]:
        _absolute(part, "bind path")
        _require("," not in part, "bind paths cannot contain commas")
    _require(len(parts) < 3 or parts[2] in {"ro", "rw"}, "bind option must be ro or rw")
    return text


@dataclass(frozen=True, slots=True, init=False)
class Apptainer:
    """Run each Task in one immutable Apptainer image with a clean environment."""

    executable: PurePosixPath
    image: PurePosixPath
    binds: tuple[str, ...] = ()

    def __init__(
        self, *, executable: PosixInput, image: PosixInput, binds: Iterable[str] = ()
    ) -> None:
        container = _absolute(image, "image")
        use = "Apptainer reads it as a URI such as docker://"
        object.__setattr__(self, "executable", _absolute(executable, "apptainer"))
        object.__setattr__(self, "image", _plain(container, "image", ":", use))
        object.__setattr__(self, "binds", tuple(_bind(item) for item in _strings(binds, "binds")))


@dataclass(frozen=True, slots=True, init=False)
class Target:
    """One concrete Slurm route with conservative request ceilings.

    ``host`` names the SSH destination; ``None`` runs scheduler commands locally, for use on a
    login node. ``container`` selects Apptainer; ``None`` runs each Task's absolute ``args[0]``
    directly under a clean environment.
    """

    slurm_bin: PurePosixPath
    work_root: PurePosixPath
    log_root: PurePosixPath
    partitions: tuple[str, ...]
    max_tasks_per_allocation: int
    max_cpus_per_allocation: int
    max_memory_mib_per_allocation: int
    max_time_limit: timedelta
    host: str | None = None
    container: Apptainer | None = None
    account: str | None = None
    qos: str | None = None
    constraint: str | None = None
    gpu_gres: str | None = None
    max_gpus_per_allocation: int = 0
    max_allocations_per_submit: int | None = None
    max_script_bytes: int = DEFAULT_MAX_SCRIPT_BYTES

    def __init__(
        self,
        *,
        slurm_bin: PosixInput,
        work_root: PosixInput,
        log_root: PosixInput,
        partitions: Iterable[str],
        max_tasks_per_allocation: int,
        max_cpus_per_allocation: int,
        max_memory_mib_per_allocation: int,
        max_time_limit: str | timedelta,
        host: str | None = None,
        container: Apptainer | None = None,
        account: str | None = None,
        qos: str | None = None,
        constraint: str | None = None,
        gpu_gres: str | None = None,
        max_gpus_per_allocation: int = 0,
        max_allocations_per_submit: int | None = None,
        max_script_bytes: int = DEFAULT_MAX_SCRIPT_BYTES,
    ) -> None:
        _require(
            host is None or (isinstance(host, str) and _HOST.fullmatch(host) is not None),
            "host must be one SSH destination ([user@]host or ssh_config alias)",
        )
        _require(container is None or isinstance(container, Apptainer), "container is invalid")
        frozen_partitions = tuple(
            _token(item, "partition") for item in _strings(partitions, "partitions")
        )
        _require(bool(frozen_partitions), "partitions must be nonempty")
        _require(len(set(frozen_partitions)) == len(frozen_partitions), "partitions must be unique")
        if gpu_gres is not None:
            _require(
                isinstance(gpu_gres, str) and _GRES.fullmatch(gpu_gres) is not None,
                "gpu_gres must be one count-free GPU resource such as gpu or gpu:a100",
            )
        gpus = _integer(max_gpus_per_allocation, "max_gpus_per_allocation", minimum=0)
        _require((gpu_gres is None) == (gpus == 0), "gpu_gres and max_gpus_per_allocation conflict")
        values: dict[str, object] = {
            "slurm_bin": _absolute(slurm_bin, "slurm_bin"),
            "work_root": _plain(
                _absolute(work_root, "work_root"),
                "work_root",
                ",:",
                "it is bound into containers as SOURCE:DESTINATION",
            ),
            "log_root": _plain(
                _absolute(log_root, "log_root"),
                "log_root",
                "%",
                "Slurm expands '%' patterns in output paths",
            ),
            "partitions": frozen_partitions,
            "max_time_limit": whole_minutes(duration(max_time_limit, "max_time_limit")),
            "host": host,
            "container": container,
            "account": _optional_token(account, "account"),
            "qos": _optional_token(qos, "qos"),
            "constraint": _optional_token(constraint, "constraint"),
            "gpu_gres": gpu_gres,
            "max_gpus_per_allocation": gpus,
        }
        positive = {
            "max_tasks_per_allocation": max_tasks_per_allocation,
            "max_cpus_per_allocation": max_cpus_per_allocation,
            "max_memory_mib_per_allocation": max_memory_mib_per_allocation,
            "max_allocations_per_submit": max_allocations_per_submit,
            "max_script_bytes": max_script_bytes,
        }
        for name, item in positive.items():
            optional = name == "max_allocations_per_submit" and item is None
            values[name] = None if optional else _integer(item, name, minimum=1)
        for name, item in values.items():
            object.__setattr__(self, name, item)


# --- Profile ---------------------------------------------------------------------------------

_CONTAINER_KEYS = frozenset({"apptainer", "image", "binds"})
_TARGET_KEYS = (
    frozenset(item.name for item in dataclasses.fields(Target) if item.name != "container")
    | _CONTAINER_KEYS
)
_RESOURCE_KEYS = frozenset(item.name for item in dataclasses.fields(Resources))
_SECTIONS = {"target": _TARGET_KEYS, "resources": _RESOURCE_KEYS}


@dataclass(frozen=True, slots=True)
class Profile:
    """A nonbinding label plus the complete Target and Resources for one plan."""

    label: str
    target: Target
    resources: Resources

    def __post_init__(self) -> None:
        _text(self.label, "profile label")
        _require(isinstance(self.target, Target), "profile target must be a Target")
        _require(isinstance(self.resources, Resources), "profile resources must be Resources")

    @classmethod
    def load(cls, path: StrPath, *, name: str | None = None) -> Self:
        """Select one profile from a ``SERVATUS.toml`` document.

        Document-level ``[target]`` and ``[resources]`` tables supply per-key defaults that each
        profile's own tables override. An explicit ``name`` overrides ``default_profile``; a sole
        profile selects itself. Unknown keys are rejected everywhere.
        """
        source = Path(path)
        try:
            with source.open("rb") as handle:
                document = cast(dict[str, object], tomllib.load(handle))
        except FileNotFoundError:
            raise NotFound(f"configuration file does not exist: {source}") from None
        except (OSError, ValueError) as error:  # ValueError: undecodable, malformed, or NUL
            raise ConfigurationError(f"cannot read TOML configuration {source}: {error}") from None
        if unknown := document.keys() - {"profiles", "default_profile", "target", "resources"}:
            raise ConfigurationError(f"unknown document keys: {', '.join(sorted(unknown))}")
        defaults = _sections(document, "the document")
        profiles: object = document.get("profiles")
        if not isinstance(profiles, dict) or not cast(dict[str, object], profiles):
            raise ConfigurationError("profiles must be a nonempty table")
        tables: dict[str, dict[str, dict[str, object]]] = {}
        for label, raw in cast(dict[str, object], profiles).items():
            _require(isinstance(raw, dict), f"profile {label!r} must be a table")
            mapping = cast(dict[str, object], raw)
            if unknown := mapping.keys() - _SECTIONS.keys():
                raise ConfigurationError(
                    f"unknown keys in profile {label!r}: {', '.join(sorted(unknown))}"
                )
            tables[label] = _sections(mapping, f"profile {label!r}")
        default = document.get("default_profile")
        if default is not None:
            _require(
                isinstance(default, str) and default in tables,
                "default_profile must name a declared profile",
            )
        selected = name if name is not None else cast(str | None, default)
        if selected is None and len(tables) == 1:
            selected = next(iter(tables))
        _require(selected is not None, "profile selection is required: pass a profile name")
        selected = cast(str, selected)
        _require(selected in tables, f"profile {selected!r} is not declared")
        merged = {
            section: {**defaults[section], **tables[selected][section]} for section in _SECTIONS
        }
        return cls(selected, _target(merged["target"]), _resources(merged["resources"]))


def _sections(mapping: dict[str, object], owner: str) -> dict[str, dict[str, object]]:
    tables: dict[str, dict[str, object]] = {}
    for section, allowed in _SECTIONS.items():
        raw = mapping.get(section, {})
        _require(isinstance(raw, dict), f"{section} in {owner} must be a table")
        table = cast(dict[str, object], raw)
        if unknown := table.keys() - allowed:
            raise ConfigurationError(
                f"unknown {section} keys in {owner}: {', '.join(sorted(unknown))}"
            )
        tables[section] = table
    return tables


def _required(cls: type[Target] | type[Resources]) -> set[str]:
    return {item.name for item in dataclasses.fields(cls) if item.default is dataclasses.MISSING}


def _missing(values: Mapping[str, object], required: Iterable[str], section: str) -> None:
    if missing := sorted(set(required) - values.keys()):
        raise ConfigurationError(f"missing {section} keys: {', '.join(missing)}")


def _target(values: dict[str, object]) -> Target:
    container_values = {name: values.pop(name) for name in _CONTAINER_KEYS & values.keys()}
    container = None
    if container_values:
        _missing(container_values, ("apptainer", "image"), "target")
        container = Apptainer(
            executable=cast(PosixInput, container_values["apptainer"]),
            image=cast(PosixInput, container_values["image"]),
            binds=cast(Iterable[str], container_values.get("binds", ())),
        )
    _missing(values, _required(Target) - {"container"}, "target")
    _require(isinstance(values["partitions"], list), "partitions must be an array of strings")
    return Target(container=container, **values)  # pyright: ignore[reportArgumentType]


def _resources(values: dict[str, object]) -> Resources:
    _missing(values, _required(Resources), "resources")
    return Resources(**values)  # pyright: ignore[reportArgumentType]
