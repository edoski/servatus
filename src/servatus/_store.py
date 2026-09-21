from __future__ import annotations

import fcntl
import os
import stat
from collections.abc import Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from ._errors import ConfigurationError, TaskConflict
from ._model import State, canonical, decode_json, decode_state, state_document

_MAX_STATE_BYTES = 256 * 1024 * 1024


@dataclass(slots=True)
class Transaction:
    descriptor: int
    state: State | None

    def commit(self, state: State) -> None:
        descriptor = self.descriptor
        encoded = canonical(state_document(state)) + b"\n"
        if len(encoded) > _MAX_STATE_BYTES:
            raise TaskConflict(
                f"campaign state is {len(encoded)} bytes; maximum is {_MAX_STATE_BYTES}"
            )
        name = f".campaign-{os.urandom(12).hex()}.tmp"
        stage = -1
        installed = False
        try:
            stage = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=descriptor,
            )
            view = memoryview(encoded)
            while view:
                written = os.write(stage, view)
                if written <= 0:
                    raise OSError("campaign state write made no progress")
                view = view[written:]
            os.fsync(stage)
            os.close(stage)
            stage = -1
            try:
                current = os.stat("campaign.json", dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                current = None
            if current is not None:
                _require_owner_file(current, "campaign.json")
            os.replace(name, "campaign.json", src_dir_fd=descriptor, dst_dir_fd=descriptor)
            installed = True
            os.fsync(descriptor)
        except BaseException:
            if stage >= 0:
                with suppress(OSError):
                    os.close(stage)
            if not installed:
                with suppress(FileNotFoundError):
                    os.unlink(name, dir_fd=descriptor)
            raise
        self.state = state


@dataclass(frozen=True, slots=True)
class Store:
    path: Path
    entry: os.stat_result

    @classmethod
    def create(cls, path: Path) -> Store:
        return cls(*_open_or_create_directory(path))

    @classmethod
    def load(cls, path: Path) -> Store:
        store = cls(*_open_existing_directory(path))
        store.read()
        return store

    def read(self) -> State:
        with self.transaction() as transaction:
            assert transaction.state is not None
            return transaction.state

    @contextmanager
    def transaction(self, *, creating: bool = False) -> Generator[Transaction]:
        descriptor = _open_campaign_directory(self.path, self.entry)
        lock_fd = -1
        try:
            lock_fd = os.open(
                ".lock",
                os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=descriptor,
            )
            _require_owner_file(os.fstat(lock_fd), ".lock")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            lock_entry = os.stat(".lock", dir_fd=descriptor, follow_symlinks=False)
            if (lock_entry.st_dev, lock_entry.st_ino) != (
                os.fstat(lock_fd).st_dev,
                os.fstat(lock_fd).st_ino,
            ):
                raise TaskConflict("campaign lock was replaced")
            try:
                state = _load_state(descriptor)
            except FileNotFoundError:
                if not creating:
                    raise TaskConflict("campaign state is missing") from None
                state = None
            yield Transaction(descriptor, state)
        finally:
            if lock_fd >= 0:
                os.close(lock_fd)
            os.close(descriptor)


def _open_or_create_directory(path: Path) -> tuple[Path, os.stat_result]:
    raw = os.fspath(path)
    if "\0" in raw:
        raise TaskConflict("campaign path contains NUL")
    normalized = Path(os.path.abspath(raw))
    name = normalized.name
    if name in {"", ".", ".."}:
        raise TaskConflict("campaign path must name one directory")
    try:
        parent_fd = os.open(
            normalized.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError as error:
        raise TaskConflict("campaign parent is unavailable or unsafe") from error
    child_fd = -1
    created = False
    try:
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
            created = True
        except FileExistsError:
            pass
        try:
            child_fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
        except OSError as error:
            raise TaskConflict("campaign directory is unavailable or unsafe") from error
        entry = os.fstat(child_fd)
        _require_owner_directory(entry)
        os.fsync(parent_fd)
        return normalized, entry
    except BaseException as error:
        if created:
            try:
                os.rmdir(name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError as cleanup_error:
                error.add_note(
                    f"Servatus could not remove unsynced campaign directory: {cleanup_error}"
                )
        raise
    finally:
        if child_fd >= 0:
            os.close(child_fd)
        os.close(parent_fd)


def _open_existing_directory(path: Path) -> tuple[Path, os.stat_result]:
    raw = os.fspath(path)
    if "\0" in raw:
        raise TaskConflict("campaign path contains NUL")
    normalized = Path(os.path.abspath(raw))
    try:
        descriptor = os.open(
            normalized, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
    except OSError as error:
        raise TaskConflict("campaign directory is unavailable or unsafe") from error
    try:
        entry = os.fstat(descriptor)
        _require_owner_directory(entry)
    finally:
        os.close(descriptor)
    return normalized, entry


def _open_campaign_directory(path: Path, expected: os.stat_result) -> int:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    entry = os.fstat(descriptor)
    try:
        _require_owner_directory(entry)
        if entry.st_dev != expected.st_dev or entry.st_ino != expected.st_ino:
            raise TaskConflict("campaign directory was replaced")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _require_owner_directory(entry: os.stat_result) -> None:
    if not stat.S_ISDIR(entry.st_mode) or entry.st_uid != os.getuid() or entry.st_mode & 0o077:
        raise TaskConflict("campaign directory must be owner-only and symlink-free")


def _require_owner_file(entry: os.stat_result, name: str) -> None:
    if not stat.S_ISREG(entry.st_mode) or entry.st_uid != os.getuid() or entry.st_mode & 0o077:
        raise TaskConflict(f"campaign {name} must be an owner-only regular file")


def _load_state(descriptor: int) -> State:
    state_fd = os.open(
        "campaign.json",
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        dir_fd=descriptor,
    )
    try:
        entry = os.fstat(state_fd)
        _require_owner_file(entry, "campaign.json")
        if entry.st_size > _MAX_STATE_BYTES:
            raise TaskConflict("campaign state is too large")
        chunks: list[bytes] = []
        remaining = entry.st_size + 1
        while remaining and (chunk := os.read(state_fd, min(remaining, 1024 * 1024))):
            chunks.append(chunk)
            remaining -= len(chunk)
        if sum(map(len, chunks)) != entry.st_size:
            raise TaskConflict("campaign state changed while reading")
    finally:
        os.close(state_fd)
    try:
        state = decode_json(b"".join(chunks))
    except (UnicodeDecodeError, ValueError) as error:
        raise TaskConflict("campaign state is invalid JSON") from error
    try:
        return decode_state(state)
    except (ValueError, TypeError, ConfigurationError) as error:
        raise TaskConflict(f"invalid campaign state: {error}") from error
