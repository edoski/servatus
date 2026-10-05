"""Identity-bound resumable private work built from immutable, fully initialized levels.

A Workspace keeps its private work in a hidden owner-only container beside the destination:
`<container>/.lock` (lifecycle lease), `<container>/work` (application work), and
`<container>/.identity` (identity digest plus the exact container, lock, and work inode pins).
A child Workspace keeps the same layout inside its root's work directory and holds a shared
lease on the root while it holds an exclusive lease on itself.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
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
from ._transaction import Draft, Publication, draft_builder, result, transact, warn_pending

_HEADER = b"servatus-workspace-v2\n"
_MAX_RECORD = len(_HEADER) + 65 + 3 * 21
_LOCK = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK


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
        if identity:
            _verify_identity(self, location)


@dataclass(frozen=True, slots=True)
class _Session:
    stack: ExitStack
    parent: Pin
    levels: tuple[_Level, ...]

    @property
    def directory(self) -> Pin:
        """The pinned directory that receives this level's publication."""
        return self.parent if len(self.levels) == 1 else self.levels[0].work


class Workspace:
    """Stable, identity-bound private work for one future destination.

    Entering acquires a nonblocking exclusive lifecycle lease (`Busy` when held elsewhere) and
    binds the hidden container to the SHA-256 digest of `identity` (`WorkspaceConflict` when it
    belongs to other work). Work survives failures and restarts until `publish` commits the
    destination and removes it, or `discard` removes it explicitly. Entering when the destination
    already exists first reclaims this identity's leftover private work, then raises
    `DestinationExists`.
    """

    __slots__ = ("_finished", "_location", "_session")

    def __init__(self, destination: StrPath, *, identity: bytes) -> None:
        if not isinstance(identity, bytes):
            raise ConfigurationError("workspace identity must be bytes")
        parent, name = _fs.split(destination)
        self._location = _Location(parent, name, _digest(identity))
        self._session: _Session | None = None
        self._finished: str | None = None

    @classmethod
    def _at(cls, location: _Location) -> Workspace:
        workspace = cls.__new__(cls)
        workspace._location = location
        workspace._session = None
        workspace._finished = None
        return workspace

    def child(self, name: str, *, identity: bytes) -> Workspace:
        """A child Workspace publishing `<self.path>/<name>`; siblings may run concurrently."""
        if self._location.root is not None:
            raise ConfigurationError("child workspaces cannot contain child workspaces")
        if not isinstance(identity, bytes):
            raise ConfigurationError("workspace identity must be bytes")
        location = self._location
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
        with ExitStack() as stack:
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
            directories = [parent, *(level.work for level in levels[:-1])]
            _verify(chain, parent, directories, levels)
            for directory, location in zip(directories, chain, strict=True):
                directory.absent(location.name, location.destination)
            self._session = _Session(stack.pop_all(), parent, tuple(levels))
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
        checked = None if mode is None else _fs.check_mode(mode)
        self._verify(session)
        location, directory = self._location, session.directory
        clean = transact(
            location.parent, directory, location.name, draft_builder(build), mode=checked
        )
        self._finished = "published"
        try:
            self._verify(session)
            _remove_level_tree(directory, location, session.levels[-1])
        except Exception:
            clean = False
        else:
            clean = _fs.try_sync(directory.fd) and clean
        return result(location.destination, clean)

    def discard(self) -> None:
        """Remove this workspace's private work (and a root's children) while entered.

        Removal is exclusive (the lease is held) and pinned: a moved or substituted entry is
        preserved and reported as `UnsafeFilesystem`.
        """
        session = self._active()
        self._verify(session)
        self._finished = "discarded"
        _remove_level_tree(session.directory, self._location, session.levels[-1])
        _fs.sync(session.directory.fd)

    def _active(self) -> _Session:
        if self._session is None:
            raise RuntimeError("workspace must be entered first")
        if self._finished is not None:
            raise RuntimeError(f"workspace was already {self._finished}")
        return self._session

    def _verify(self, session: _Session) -> None:
        chain = self._location.chain
        directories = [session.parent, *(level.work for level in session.levels[:-1])]
        _verify(chain, session.parent, directories, session.levels)


def _verify(
    chain: tuple[_Location, ...],
    parent: Pin,
    directories: list[Pin],
    levels: list[_Level] | tuple[_Level, ...],
) -> None:
    parent.verify_path(chain[0].parent)
    for directory, level, location in zip(directories, levels, chain, strict=True):
        level.verify(directory, location)


def _digest(identity: bytes) -> bytes:
    return hashlib.sha256(identity).hexdigest().encode("ascii") + b"\n"


@contextmanager
def _coordinate(directory: Pin) -> Generator[None]:
    """Short exclusive parent coordination; never held across a durability sync."""
    fcntl.flock(directory.fd, fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(directory.fd, fcntl.LOCK_UN)


def _remove_level_tree(directory: Pin, location: _Location, level: _Level) -> None:
    with _coordinate(directory):
        _fs.remove(directory, location.container, level.container.entry, pinned=level.container)


def _open_lock(container: Pin) -> Pin:
    try:
        fd = os.open(".lock", _LOCK, 0o600, dir_fd=container.fd)
    except OSError as error:
        raise UnsafeFilesystem("workspace lock is unsafe") from error
    lock = _fs.adopt(fd, ".lock", device=container.entry.st_dev, regular=True)
    try:
        _fs.require_owner_only(lock.entry, "workspace lock .lock")
        container.expect(".lock", lock.entry)
    except BaseException:
        lock.close()
        raise
    return lock


def _lease(lock: Pin, location: _Location, *, exclusive: bool) -> None:
    try:
        fcntl.flock(lock.fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
    except OSError as error:
        if error.errno in {errno.EACCES, errno.EAGAIN}:
            raise Busy(f"workspace is already in use: {location.destination}") from error
        raise


def _open_level(stack: ExitStack, parent: Pin, location: _Location, *, exclusive: bool) -> _Level:
    existing: DestinationExists | None = None
    reclaimed = False
    with _coordinate(parent):
        container = stack.enter_context(parent.mkdir(location.container, exist_ok=True))
        _fs.require_owner_only(container.entry, f"workspace container {location.container}")
        lock = stack.enter_context(_open_lock(container))
        _lease(lock, location, exclusive=exclusive)
        if parent.lstat(location.name) is not None:  # published before this lease was acquired
            existing = DestinationExists(f"destination already exists: {location.destination}")
            with suppress(Busy):
                if not exclusive:  # reclaim only if no sibling child holds the shared lease
                    _lease(lock, location, exclusive=True)
                reclaimed = _remove_leftover(parent, location, container, lock, existing)
    if existing is not None:
        if reclaimed:
            _fs.try_sync(parent.fd)
        raise existing
    # The lease now excludes every compliant remover, so the work directory is stable.
    work = stack.enter_context(container.mkdir("work", exist_ok=True))
    level = _Level(container, lock, work)
    level.verify(parent, location, identity=False)
    if container.lstat(".identity") is None:
        for pin in (lock, work, container, parent):
            _fs.sync(pin.fd)
        _install_identity(level, location)
    level.verify(parent, location)
    return level


def _reclaim(parent: Pin, location: _Location, existing: DestinationExists) -> None:
    """Best effort: remove this identity's private work left beside a published destination."""
    reclaimed = False
    try:
        with ExitStack() as stack, _coordinate(parent):
            if parent.lstat(location.container) is None:
                return
            container = stack.enter_context(parent.open(location.container))
            _fs.require_owner_only(container.entry, f"workspace container {location.container}")
            lock = stack.enter_context(_open_lock(container))
            _lease(lock, location, exclusive=True)
            reclaimed = _remove_leftover(parent, location, container, lock, existing)
    except Exception as error:
        existing.add_note(f"Servatus could not reclaim private work at {location.path}: {error}")
        return
    if reclaimed:
        _fs.try_sync(parent.fd)


def _remove_leftover(
    parent: Pin, location: _Location, container: Pin, lock: Pin, existing: DestinationExists
) -> bool:
    """Remove a leased container that is bound to this identity or has no identity.

    Without an identity the container was never exposed to the application (initialization
    stopped early) or is the remainder of an interrupted post-publication cleanup.
    """
    try:
        if container.lstat(".identity") is not None:
            with container.open("work") as work:
                _Level(container, lock, work).verify(parent, location)
        _fs.remove(parent, location.container, container.entry, pinned=container)
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
        _fs.discard(container, stage, entry) and _fs.try_sync(container.fd)
    ):
        warn_pending(
            "workspace identity was installed, but identity-stage cleanup remains pending",
            stacklevel=4,
        )


def _verify_identity(level: _Level, location: _Location) -> None:
    try:
        record = level.container.open(".identity", directory=False)
    except UnsafeFilesystem as error:
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
        data = b""
        while len(data) <= record.entry.st_size and (chunk := os.read(record.fd, _MAX_RECORD + 1)):
            data += chunk
        prefix = _HEADER + location.identity
        if not data.startswith(prefix):
            raise WorkspaceConflict(
                f"private work at {location.path} belongs to a different identity "
                f"than this workspace for {location.destination}"
            )
        if data[len(prefix) :] != level.record:
            raise UnsafeFilesystem(f"workspace lifecycle entries changed: {location.path}")
        level.container.expect(".identity", record.entry)
