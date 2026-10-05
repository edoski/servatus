"""The durable Campaign store: one owner-only directory, one lock, one canonical state file.

Layout::

    <campaign>/            owner-only directory (0700, owned by the effective user)
        .lock              owner-only regular file; ``flock`` serializes every transaction
        campaign.json      canonical schema-7 state, replaced atomically on each change

Every operation reopens the directory without following a final symlink and checks that it is
still the directory the store was opened on. Reads hold a shared lock and writes an exclusive one;
the lock file's inode is re-checked after locking so a replaced lock cannot split writers.

Decoding is the expensive step, so each ``Store`` caches the last state with the exact bytes it
came from. A read compares the file's bytes with the cache and decodes only when they differ.
Commits fill the cache with the bytes they wrote.
"""

from __future__ import annotations

import errno
import fcntl
import os
import stat
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from pathlib import Path

from ..errors import ConfigurationError, Conflict, CorruptState, NotFound, UnsafeFilesystem
from ._config import StrPath
from ._state import State, decode, encode

MAX_STATE_BYTES = 256 * 1024 * 1024
STATE_NAME = "campaign.json"
LOCK_NAME = ".lock"
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
_READ_CHUNK = 1024 * 1024


class Store:
    """A locked, durable, content-cached home for one Campaign ``State``."""

    __slots__ = ("_cache", "_identity", "path")

    def __init__(self, path: Path, identity: tuple[int, int]) -> None:
        self.path = path
        self._identity = identity
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
        if parents:
            os.makedirs(location.parent, exist_ok=True)
        try:
            parent = os.open(location.parent, _DIRECTORY_FLAGS)
        except FileNotFoundError:
            raise NotFound(f"campaign parent directory does not exist: {location.parent}") from None
        except OSError as error:
            raise UnsafeFilesystem(f"campaign parent is not a usable directory: {error}") from None
        try:
            try:
                os.mkdir(location.name, 0o700, dir_fd=parent)
            except FileExistsError:
                pass
            else:
                os.fsync(parent)
            descriptor = _open_directory(location.name, parent)
        finally:
            os.close(parent)
        try:
            entry = os.fstat(descriptor)
            store = cls(location, (entry.st_dev, entry.st_ino))
            with _locked(descriptor, fcntl.LOCK_EX):
                try:
                    os.stat(STATE_NAME, dir_fd=descriptor, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise Conflict(f"campaign already exists: {location}")
                store._commit(descriptor, state)
        finally:
            os.close(descriptor)
        return store

    @classmethod
    def open(cls, path: StrPath) -> Store:
        """Open an existing campaign and validate its state. ``NotFound`` if there is none."""
        location = _location(path)
        try:
            descriptor = _open_directory(location)
        except NotFound:
            raise NotFound(f"campaign does not exist: {location}") from None
        try:
            entry = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        store = cls(location, (entry.st_dev, entry.st_ino))
        store.read()
        return store

    def read(self) -> State:
        """Return the current state, decoding only if its bytes changed since the last call."""
        with self._transaction(fcntl.LOCK_SH) as descriptor:
            return self._load(descriptor)

    def update(self, change: Callable[[State], State]) -> State:
        """Apply ``change`` to the current state under the exclusive lock and commit the result.

        Returning the same object commits nothing. Exceptions from ``change`` propagate and leave
        the stored state untouched.
        """
        with self._transaction(fcntl.LOCK_EX) as descriptor:
            current = self._load(descriptor)
            changed = change(current)
            if changed is current:
                return current
            if not isinstance(changed, State):
                raise TypeError("a campaign state change must return a State")
            self._commit(descriptor, changed)
            return changed

    # --- internals ----------------------------------------------------------------------------

    @contextmanager
    def _transaction(self, operation: int) -> Generator[int]:
        descriptor = _open_directory(self.path)
        try:
            entry = os.fstat(descriptor)
            if (entry.st_dev, entry.st_ino) != self._identity:
                raise UnsafeFilesystem(f"campaign directory was replaced: {self.path}")
            with _locked(descriptor, operation):
                yield descriptor
        finally:
            os.close(descriptor)

    def _load(self, descriptor: int) -> State:
        data = _read_state(descriptor)
        cache = self._cache
        if cache is not None and cache[0] == data:
            return cache[1]
        state = decode(data)
        self._cache = (data, state)
        return state

    def _commit(self, descriptor: int, state: State) -> None:
        data = encode(state)
        if len(data) > MAX_STATE_BYTES:
            raise Conflict(
                f"campaign state would be {len(data)} bytes; the limit is {MAX_STATE_BYTES}"
            )
        replace_file(descriptor, STATE_NAME, data)
        self._cache = (data, state)


def replace_file(dir_fd: int, name: str, data: bytes, *, mode: int = 0o600) -> None:
    """Atomically replace ``name`` in ``dir_fd`` with ``data`` and make the change durable.

    The bytes go to a fresh exclusive stage file that is fully written and synced, renamed over
    ``name``, and followed by a directory sync. Before the rename, any failure removes the stage
    and leaves ``name`` untouched; after it, the new content is visible even if the directory
    sync fails.
    """
    stage = f".{name}.{os.urandom(8).hex()}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    renamed = False
    try:
        handle = os.open(stage, flags, mode, dir_fd=dir_fd)
        try:
            view = memoryview(data)
            while view:
                written = os.write(handle, view)
                if written <= 0:
                    raise OSError(errno.EIO, "campaign state write made no progress")
                view = view[written:]
            os.fsync(handle)
        finally:
            os.close(handle)
        os.rename(stage, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        renamed = True
        os.fsync(dir_fd)
    finally:
        if not renamed:
            with suppress(FileNotFoundError):
                os.unlink(stage, dir_fd=dir_fd)


def _location(path: StrPath) -> Path:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\0" in raw:
        raise ConfigurationError("campaign path must be a nonempty text path without NUL")
    location = Path(os.path.abspath(raw))
    if location.name in {"", ".", ".."}:
        raise ConfigurationError(f"campaign path must name one directory: {raw}")
    return location


def _open_directory(path: StrPath, dir_fd: int | None = None) -> int:
    try:
        descriptor = os.open(path, _DIRECTORY_FLAGS, dir_fd=dir_fd)
    except FileNotFoundError:
        raise NotFound(f"campaign directory does not exist: {path}") from None
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR, errno.EACCES, errno.EPERM):
            raise UnsafeFilesystem(f"campaign path is not a plain directory: {path}") from None
        raise
    try:
        _require_owner_only(os.fstat(descriptor), stat.S_ISDIR, "campaign directory")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _require_owner_only(entry: os.stat_result, kind: Callable[[int], bool], what: str) -> None:
    if not kind(entry.st_mode) or entry.st_uid != os.geteuid() or entry.st_mode & 0o077:
        raise UnsafeFilesystem(f"{what} must be owner-only and owned by the current user")


@contextmanager
def _locked(descriptor: int, operation: int) -> Generator[None]:
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        lock = os.open(LOCK_NAME, flags, 0o600, dir_fd=descriptor)
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.EISDIR, errno.EACCES, errno.EPERM):
            raise UnsafeFilesystem("campaign lock is not a plain file") from None
        raise
    try:
        _require_owner_only(os.fstat(lock), stat.S_ISREG, "campaign lock")
        fcntl.flock(lock, operation)
        held = os.fstat(lock)
        try:
            named = os.stat(LOCK_NAME, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            named = None
        if named is None or (named.st_dev, named.st_ino) != (held.st_dev, held.st_ino):
            raise UnsafeFilesystem("campaign lock was replaced while waiting for it")
        yield
    finally:
        os.close(lock)


def _read_state(descriptor: int) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        handle = os.open(STATE_NAME, flags, dir_fd=descriptor)
    except FileNotFoundError:
        raise NotFound("campaign state does not exist") from None
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.EACCES, errno.EPERM):
            raise UnsafeFilesystem("campaign state is not a plain file") from None
        raise
    try:
        entry = os.fstat(handle)
        _require_owner_only(entry, stat.S_ISREG, "campaign state")
        if entry.st_size > MAX_STATE_BYTES:
            raise CorruptState(
                f"campaign state is {entry.st_size} bytes; the limit is {MAX_STATE_BYTES}"
            )
        chunks: list[bytes] = []
        remaining = entry.st_size + 1
        while remaining and (chunk := os.read(handle, min(remaining, _READ_CHUNK))):
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(handle)
    data = b"".join(chunks)
    if len(data) != entry.st_size:
        raise UnsafeFilesystem("campaign state changed while it was being read")
    return data
