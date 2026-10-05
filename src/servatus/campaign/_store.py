"""The durable Campaign store: one owner-only directory, one lock, one canonical state file.

Layout::

    <campaign>/            owner-only directory (0700, owned by the effective user)
        .lock              owner-only regular file; ``flock`` serializes every transaction
        campaign.json      canonical schema-7 state, replaced atomically on each change

Every operation reopens the directory without following a final symlink and checks that it is
still the directory the store was opened on. Reads hold a shared lock and writes an exclusive one;
the lock file's inode is re-checked after locking so a replaced lock cannot split writers. Missing
entries are ``NotFound``; other filesystem failures map like publication's (see ``_fs.os_error``).

Decoding is the expensive step, so each ``Store`` caches the last state with the exact bytes it
came from. A read compares the file's bytes with the cache and decodes only when they differ.
Commits fill the cache with the bytes they wrote.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from .. import _fs
from .._fs import Pin
from ..errors import ConfigurationError, Conflict, CorruptState, NotFound, UnsafeFilesystem
from ._config import StrPath
from ._state import State, decode, encode

MAX_STATE_BYTES = 256 * 1024 * 1024
STATE_NAME = "campaign.json"
LOCK_NAME = ".lock"


class Store:
    """A locked, durable, content-cached home for one Campaign ``State``."""

    __slots__ = ("_cache", "_entry", "path")

    def __init__(self, path: Path, entry: os.stat_result) -> None:
        self.path = path
        self._entry = entry
        self._cache: tuple[bytes, State] | None = None

    @classmethod
    def create(cls, path: StrPath, state: State, *, parents: bool = False) -> Store:
        """Create the directory if needed and write the first state.

        An existing owner-only directory without state is reused (an earlier create may have
        stopped before writing). ``Conflict`` if state already exists; ``NotFound`` if the parent
        is missing and ``parents`` is false.
        """
        if not isinstance(state, State):
            raise ConfigurationError("state must be a campaign State")
        location = _location(path)
        with _fs.Boundary(location, missing=NotFound), ExitStack() as stack:
            if parents:
                os.makedirs(location.parent, exist_ok=True)
            parent = stack.enter_context(
                _open(
                    location.parent, f"campaign parent directory does not exist: {location.parent}"
                )
            )
            try:
                os.mkdir(location.name, 0o700, dir_fd=parent.fd)
            except FileExistsError:
                pass
            else:
                _fs.sync(parent.fd)
            directory = stack.enter_context(parent.open(location.name, same_device=False))
            _fs.require_owner_only(directory.entry, "campaign directory")
            store = cls(location, directory.entry)
            with _locked(directory, fcntl.LOCK_EX):
                if directory.lstat(STATE_NAME) is not None:
                    raise Conflict(f"campaign already exists: {location}")
                store._commit(directory, state)
        return store

    @classmethod
    def open(cls, path: StrPath) -> Store:
        """Open an existing campaign and validate its state. ``NotFound`` if there is none."""
        location = _location(path)
        with _fs.Boundary(location, missing=NotFound), _directory(location) as directory:
            store = cls(location, directory.entry)
        store.read()
        return store

    def read(self) -> State:
        """Return the current state, decoding only if its bytes changed since the last call."""
        with _fs.Boundary(self.path, missing=NotFound), self._transaction(fcntl.LOCK_SH) as pin:
            return self._load(pin)

    def update(self, change: Callable[[State], State]) -> State:
        """Apply ``change`` to the current state under the exclusive lock and commit the result.

        Returning the same object commits nothing. Exceptions from ``change`` propagate and leave
        the stored state untouched. The lock is not reentrant: ``change`` must not use any
        ``Store`` for this campaign, or it deadlocks.
        """
        with (
            _fs.Boundary(self.path, missing=NotFound) as boundary,
            self._transaction(fcntl.LOCK_EX) as directory,
        ):
            current = self._load(directory)
            changed = boundary.passthrough(change)(current)
            if changed is current:
                return current
            if not isinstance(changed, State):
                raise TypeError("a campaign state change must return a State")
            self._commit(directory, changed)
            return changed

    # --- internals ----------------------------------------------------------------------------

    @contextmanager
    def _transaction(self, operation: int) -> Generator[Pin]:
        with _directory(self.path) as directory:
            if not _fs.same(directory.entry, self._entry):
                raise UnsafeFilesystem(f"campaign directory was replaced: {self.path}")
            with _locked(directory, operation):
                yield directory

    def _load(self, directory: Pin) -> State:
        if directory.lstat(STATE_NAME) is None:
            raise NotFound("campaign state does not exist")
        with directory.open(STATE_NAME, directory=False) as file:
            _fs.require_owner_only(file.entry, "campaign state")
            if file.entry.st_size > MAX_STATE_BYTES:
                raise CorruptState(
                    f"campaign state is {file.entry.st_size} bytes; the limit is {MAX_STATE_BYTES}"
                )
            data = _fs.read(file)
        cache = self._cache
        if cache is not None and cache[0] == data:
            return cache[1]
        state = decode(data)
        self._cache = (data, state)
        return state

    def _commit(self, directory: Pin, state: State) -> None:
        data = encode(state)
        if len(data) > MAX_STATE_BYTES:
            raise Conflict(
                f"campaign state would be {len(data)} bytes; the limit is {MAX_STATE_BYTES}"
            )
        _fs.replace_file(directory.fd, STATE_NAME, data)
        self._cache = (data, state)


def _location(path: StrPath) -> Path:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\0" in raw:
        raise ConfigurationError("campaign path must be a nonempty text path without NUL")
    location = Path(os.path.abspath(raw))
    if location.name in {"", ".", ".."}:
        raise ConfigurationError(f"campaign path must name one directory: {raw}")
    return location


def _open(path: Path, missing: str) -> Pin:
    try:
        return _fs.open_path(path)
    except FileNotFoundError as error:
        raise NotFound(missing) from error


def _directory(path: Path) -> Pin:
    """Pin the owner-only campaign directory at `path`."""
    directory = _open(path, f"campaign does not exist: {path}")
    try:
        _fs.require_owner_only(directory.entry, "campaign directory")
    except BaseException:
        directory.close()
        raise
    return directory


def _locked(directory: Pin, operation: int) -> Pin:
    return _fs.lock_file(directory, LOCK_NAME, operation, "campaign lock")
