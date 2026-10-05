"""Identity-bound resumable private work built from immutable, fully initialized levels.

A Workspace keeps its private work in a hidden owner-only container beside the destination:
`<container>/.lock` (lifecycle lease), `<container>/work` (application work), and
`<container>/.identity` (identity digest plus the exact container, lock, and work inode pins).
A child Workspace keeps the same layout inside its root's work directory and holds a shared
lease on the root while it holds an exclusive lease on itself.

Containers are removed `work` first and `.identity` last, so an interrupted removal never leaves
work without the identity that owns it. A container whose `.identity` survives its `work` is
finished off by the next opener with that identity; one with work but no identity is adopted
only while its work is still empty (initialization stops before work is exposed).
"""

from __future__ import annotations

import fcntl
import hashlib
import os
from collections.abc import Callable
from contextlib import ExitStack, suppress
from dataclasses import dataclass, replace
from pathlib import Path

from .. import _fs
from .._fs import Pin, StrPath
from ..errors import (
    Busy,
    ConfigurationError,
    DestinationExists,
    UnsafeFilesystem,
    WorkspaceConflict,
)
from ._transaction import (
    Draft,
    Publication,
    checked_mode,
    draft_builder,
    result,
    transact,
    warn_pending,
)

_HEADER = b"servatus-workspace-v2\n"
_MAX_RECORD = len(_HEADER) + 65 + 3 * 21


def _removal_order(name: str) -> tuple[int, str]:
    """`work` first and `.identity` last; everything else in between."""
    return {"work": 0, ".identity": 2}.get(name, 1), name


@dataclass(frozen=True, slots=True)
class _Location:
    """Where one level lives. A child's parent is its root's work directory."""

    parent: Path
    name: str
    identity: bytes
    root: _Location | None = None

    @property
    def destination(self) -> Path:
        return self.parent / self.name

    @property
    def container(self) -> str:
        return f".servatus-{hashlib.sha256(os.fsencode(self.name)).hexdigest()[:24]}.work"

    @property
    def path(self) -> Path:
        return self.parent / self.container / "work"

    @property
    def chain(self) -> tuple[_Location, ...]:
        return (self,) if self.root is None else (self.root, self)


@dataclass(frozen=True, slots=True)
class _Level:
    container: Pin
    lock: Pin
    work: Pin

    @property
    def record(self) -> bytes:
        pins = (self.container, self.lock, self.work)
        return b"".join(b"%d\n" % pin.entry.st_ino for pin in pins)

    def verify(self, parent: Pin, location: _Location, *, identity: bool = True) -> None:
        for directory, name, pin in (
            (parent, location.container, self.container),
            (self.container, ".lock", self.lock),
            (self.container, "work", self.work),
        ):
            _fs.require_owner_only(os.fstat(pin.fd), f"workspace entry {name}")
            directory.expect(name, pin.entry)
        if identity and _pins(self.container, location) != self.record:
            raise UnsafeFilesystem(f"workspace lifecycle entries changed: {location.path}")


@dataclass(frozen=True, slots=True)
class _Session:
    stack: ExitStack
    directories: tuple[Pin, ...]
    """The directory holding each level: the destination parent, then (for a child) root work."""
    levels: tuple[_Level, ...]

    @property
    def directory(self) -> Pin:
        """The pinned directory that receives this level's publication."""
        return self.directories[-1]

    def verify(self, chain: tuple[_Location, ...]) -> None:
        self.directories[0].verify_path(chain[0].parent)
        for directory, level, location in zip(self.directories, self.levels, chain, strict=True):
            level.verify(directory, location)


class Workspace:
    """Stable, identity-bound private work for one future destination.

    Entering acquires a nonblocking exclusive lifecycle lease (`Busy` when held elsewhere) and
    binds the hidden container to the SHA-256 digest of `identity` (`WorkspaceConflict` when it
    belongs to other work, or holds work without any identity). Work survives failures and
    restarts until `publish` commits the destination and removes it, or `discard` removes it
    explicitly. Entering when the destination already exists first reclaims this identity's
    leftover private work, then raises `DestinationExists`.
    """

    __slots__ = ("_finished", "_location", "_session")

    def __init__(self, destination: StrPath, *, identity: bytes) -> None:
        digest = _digest(identity)
        parent, name = _fs.split(destination)
        self._bind(_Location(parent, name, digest))

    @classmethod
    def _at(cls, location: _Location) -> Workspace:
        workspace = cls.__new__(cls)
        workspace._bind(location)
        return workspace

    def _bind(self, location: _Location) -> None:
        self._location = location
        self._session: _Session | None = None
        self._finished: str | None = None

    def child(self, name: str, *, identity: bytes) -> Workspace:
        """A child Workspace publishing `<self.path>/<name>`; siblings may run concurrently."""
        location = self._location
        if location.root is not None:
            raise ConfigurationError("child workspaces cannot contain child workspaces")
        return Workspace._at(_Location(location.path, _fs.leaf(name), _digest(identity), location))

    @property
    def path(self) -> Path:
        """The private work directory. Valid for the application only while entered."""
        return self._location.path

    def __enter__(self) -> Workspace:
        if self._session is not None:
            raise RuntimeError("workspace is already entered")
        _fs.require_supported_platform()
        chain = self._location.chain
        root = chain[0]
        with _fs.Boundary(self._location.destination), ExitStack() as stack:
            parent = stack.enter_context(_fs.open_path(root.parent))
            parent.verify_path(root.parent)
            if parent.lstat(root.name) is not None:
                error = DestinationExists(f"destination already exists: {root.destination}")
                if root is self._location:
                    _reclaim(parent, root, error)
                raise error
            levels: list[_Level] = []
            for location in chain:
                directory = levels[-1].work if levels else parent
                exclusive = location is self._location
                levels.append(_open_level(stack, directory, location, exclusive=exclusive))
            session = _Session(stack, (parent, *(level.work for level in levels[:-1])), (*levels,))
            session.verify(chain)
            for directory, location in zip(session.directories, chain, strict=True):
                directory.absent(location.name, location.destination)
            self._session = replace(session, stack=stack.pop_all())
        self._finished = None
        return self

    def __exit__(self, *_: object) -> None:
        session, self._session = self._session, None
        if session is not None:
            session.stack.close()

    def publish(self, build: Callable[[Draft], object], *, mode: int | None = None) -> Publication:
        """Commit a directory built from this work as the destination, then remove the work.

        A builder or commit failure preserves the work for another attempt. Directory modes
        follow `publish`. Failure to remove the work after commit sets `cleanup_pending`.
        """
        session = self._active()
        checked = checked_mode(mode)
        location, directory = self._location, session.directory
        with _fs.Boundary(location.destination) as boundary:
            session.verify(location.chain)
            builder = draft_builder(boundary.passthrough(build))
            clean = transact(location.parent, directory, location.name, builder, mode=checked)
            self._finished = "published"
            removed = _fs.succeeded(session.verify, location.chain) and _fs.succeeded(
                _remove_level, session, location
            )
            clean = removed and _fs.succeeded(_fs.sync, directory.fd) and clean
        return result(location.destination, clean)

    def discard(self) -> None:
        """Remove this workspace's private work (and a root's children) while entered.

        Removal is exclusive (the lease is held) and pinned: a moved or substituted entry is
        preserved and reported as `UnsafeFilesystem`. The work goes first and the identity
        last, so an interrupted discard is finished by the next entry with this identity.
        """
        session = self._active()
        with _fs.Boundary(self._location.path):
            session.verify(self._location.chain)
            self._finished = "discarded"
            _remove_level(session, self._location)
            _fs.sync(session.directory.fd)

    def _active(self) -> _Session:
        if self._session is None:
            raise RuntimeError("workspace must be entered first")
        if self._finished is not None:
            raise RuntimeError(f"workspace was already {self._finished}")
        return self._session


def _digest(identity: bytes) -> bytes:
    if not isinstance(identity, bytes):
        raise ConfigurationError("workspace identity must be bytes")
    return hashlib.sha256(identity).hexdigest().encode("ascii") + b"\n"


def _remove_level(session: _Session, location: _Location) -> None:
    with _fs.locked(session.directory):
        _remove_container(session.directory, location, session.levels[-1].container)


def _remove_container(directory: Pin, location: _Location, container: Pin) -> None:
    _fs.remove(directory, location.container, container, order=_removal_order)


def _open_lock(container: Pin, location: _Location, *, exclusive: bool) -> Pin:
    try:
        return _fs.lock_file(container, ".lock", _lease_mode(exclusive), "workspace lock .lock")
    except BlockingIOError as error:
        raise Busy(f"workspace is already in use: {location.destination}") from error


def _upgrade(lock: Pin, location: _Location) -> None:
    try:
        fcntl.flock(lock.fd, _lease_mode(True))
    except BlockingIOError as error:
        raise Busy(f"workspace is already in use: {location.destination}") from error


def _lease_mode(exclusive: bool) -> int:
    return (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB


def _open_level(stack: ExitStack, parent: Pin, location: _Location, *, exclusive: bool) -> _Level:
    existing: DestinationExists | None = None
    reclaimed = False
    with _fs.locked(parent):
        container, lock = _claim(stack, parent, location, exclusive=exclusive)
        if parent.lstat(location.name) is not None:  # published before this lease was acquired
            existing = DestinationExists(f"destination already exists: {location.destination}")
            with suppress(Busy):
                if not exclusive:  # reclaim only if no sibling child holds the shared lease
                    _upgrade(lock, location)
                reclaimed = _remove_leftover(parent, location, container, lock, existing)
    if existing is not None:
        if reclaimed:
            _fs.succeeded(_fs.sync, parent.fd)
        raise existing
    # The lease now excludes every compliant remover, so the work directory is stable.
    work = stack.enter_context(container.mkdir("work", exist_ok=True))
    level = _Level(container, lock, work)
    level.verify(parent, location, identity=False)
    if container.lstat(".identity") is None:
        if os.listdir(work.fd) and container.lstat(".identity") is None:
            raise WorkspaceConflict(
                f"private work at {location.path} has no workspace identity; inspect it and "
                f"remove it before starting {location.destination}"
            )
        for pin in (lock, work, container, parent):
            _fs.sync(pin.fd)
        _install_identity(level, location)
    level.verify(parent, location)
    return level


def _claim(
    stack: ExitStack, parent: Pin, location: _Location, *, exclusive: bool
) -> tuple[Pin, Pin]:
    """Pin and lease this level's container while `parent` is locked.

    A container whose `.identity` outlived its `work` is the remainder of an interrupted removal:
    this identity finishes the removal and starts afresh; any other identity is refused.
    """
    for _ in range(2):
        with ExitStack() as attempt:
            container = attempt.enter_context(parent.mkdir(location.container, exist_ok=True))
            _fs.require_owner_only(container.entry, f"workspace container {location.container}")
            lock = attempt.enter_context(_open_lock(container, location, exclusive=exclusive))
            if container.lstat("work") is not None or container.lstat(".identity") is None:
                stack.enter_context(attempt.pop_all())
                return container, lock
            if not exclusive:
                _upgrade(lock, location)
            _pins(container, location)
            _remove_container(parent, location, container)
    raise UnsafeFilesystem(f"workspace container keeps reappearing: {location.path}")


def _reclaim(parent: Pin, location: _Location, existing: DestinationExists) -> None:
    """Best effort: remove this identity's private work left beside a published destination."""
    reclaimed = False
    try:
        with ExitStack() as stack, _fs.locked(parent):
            if parent.lstat(location.container) is None:
                return
            container = stack.enter_context(parent.open(location.container))
            _fs.require_owner_only(container.entry, f"workspace container {location.container}")
            lock = stack.enter_context(_open_lock(container, location, exclusive=True))
            reclaimed = _remove_leftover(parent, location, container, lock, existing)
    except Exception as error:
        existing.add_note(f"Servatus could not reclaim private work at {location.path}: {error}")
        return
    if reclaimed:
        _fs.succeeded(_fs.sync, parent.fd)


def _remove_leftover(
    parent: Pin, location: _Location, container: Pin, lock: Pin, existing: DestinationExists
) -> bool:
    """Remove a leased container that provably belongs to this identity or holds no work.

    A bound container is verified (with its pins while `work` exists). An unbound one is
    removed only while its work is absent or empty: initialization stopped before exposure.
    """
    try:
        work = container.lstat("work")
        if container.lstat(".identity") is None:
            if work is not None:
                with container.open("work", expected=work) as pinned:
                    if os.listdir(pinned.fd):
                        raise WorkspaceConflict("the private work has no workspace identity")
        elif work is None:
            _pins(container, location)
        else:
            with container.open("work", expected=work) as pinned:
                _Level(container, lock, pinned).verify(parent, location)
        _remove_container(parent, location, container)
    except Exception as error:
        existing.add_note(f"Servatus kept private work at {location.path}: {error}")
        return False
    return True


def _install_identity(level: _Level, location: _Location) -> None:
    container = level.container
    stage = f".identity-{os.urandom(12).hex()}.tmp"
    entry = _fs.write_new(container.fd, stage, _HEADER + location.identity + level.record)
    try:
        # A concurrent opener may install it first; verification decides whether it matches.
        with suppress(DestinationExists):
            _fs.commit(container, stage, container, ".identity", entry)
        _fs.sync(container.fd)
    except BaseException as error:
        if not _fs.discard(container, stage, entry):
            error.add_note(f"Servatus could not remove identity stage {stage}")
        raise
    if container.lstat(stage) is not None and not (
        _fs.discard(container, stage, entry) and _fs.succeeded(_fs.sync, container.fd)
    ):
        warn_pending(
            "workspace identity was installed, but identity-stage cleanup remains pending",
            stacklevel=4,
        )


def _pins(container: Pin, location: _Location) -> bytes:
    """Return the inode pins recorded in `.identity` after proving it names this identity."""
    try:
        record = container.open(".identity", directory=False)
    except (UnsafeFilesystem, ConfigurationError) as error:
        raise WorkspaceConflict(
            f"workspace identity is unavailable for {location.destination}; "
            f"private work is at {location.path}"
        ) from error
    with record:
        if record.entry.st_size > _MAX_RECORD:
            raise WorkspaceConflict(
                f"workspace identity is invalid for {location.destination}; "
                f"private work is at {location.path}"
            )
        _fs.require_owner_only(record.entry, "workspace identity .identity")
        data = _fs.read(record)
        container.expect(".identity", record.entry)
    prefix = _HEADER + location.identity
    if not data.startswith(prefix):
        raise WorkspaceConflict(
            f"private work at {location.path} belongs to a different identity "
            f"than this workspace for {location.destination}"
        )
    return data[len(prefix) :]
