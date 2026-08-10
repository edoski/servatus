from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import stat
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from . import _posix
from ._errors import (
    CrossDevicePublication,
    DestinationExists,
    UnsafePublication,
    WorkConflict,
    WorkspaceBusy,
)


@dataclass(frozen=True, slots=True)
class Publication:
    destination: Path
    cleanup_pending: bool


class Draft:
    def __init__(self, path: Path, descriptor: int, device: int) -> None:
        self._path = path
        self._descriptor = descriptor
        self._device = device

    @property
    def path(self) -> Path:
        return self._path

    def link(self, source: Path, destination: str | PurePosixPath) -> None:
        components = _safe_components(destination)
        source_path = Path(source)
        try:
            source_entry = source_path.stat(follow_symlinks=False)
        except OSError as error:
            raise UnsafePublication(f"hard-link source is unavailable: {source_path}") from error
        if not stat.S_ISREG(source_entry.st_mode):
            raise UnsafePublication(f"hard-link source is not a regular file: {source_path}")
        try:
            source_fd = os.open(
                source_path,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            )
        except OSError as error:
            raise UnsafePublication(f"unsafe hard-link source: {source_path}") from error
        try:
            opened_source = os.fstat(source_fd)
            if not stat.S_ISREG(opened_source.st_mode) or not _posix.same_entry(
                opened_source, source_entry
            ):
                raise UnsafePublication(f"hard-link source changed: {source_path}")
            if opened_source.st_dev != self._device:
                raise CrossDevicePublication(
                    f"hard-link source is on another filesystem: {source_path}"
                )
            self._link_open_source(source_path, opened_source, components)
        finally:
            os.close(source_fd)

    def _link_open_source(
        self,
        source: Path,
        source_entry: os.stat_result,
        components: tuple[str, ...],
    ) -> None:
        current_fd = os.dup(self._descriptor)
        try:
            for component in components[:-1]:
                with suppress(FileExistsError):
                    os.mkdir(component, mode=0o700, dir_fd=current_fd)
                child_fd = _posix.open_directory_at(current_fd, component)
                os.close(current_fd)
                current_fd = child_fd
            leaf = components[-1]
            try:
                os.link(source, leaf, dst_dir_fd=current_fd, follow_symlinks=False)
            except FileExistsError as error:
                raise DestinationExists(
                    f"draft path already exists: {PurePosixPath(*components)}"
                ) from error
            except OSError as error:
                if error.errno == errno.EXDEV:
                    raise CrossDevicePublication(
                        f"hard-link source is on another filesystem: {source}"
                    ) from error
                raise
            linked = os.stat(leaf, dir_fd=current_fd, follow_symlinks=False)
            if not stat.S_ISREG(linked.st_mode) or not _posix.same_entry(linked, source_entry):
                with suppress(OSError):
                    os.unlink(leaf, dir_fd=current_fd)
                raise UnsafePublication(f"hard-link source changed: {source}")
        finally:
            os.close(current_fd)


def _safe_components(destination: str | PurePosixPath) -> tuple[str, ...]:
    path = PurePosixPath(destination)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise UnsafePublication(f"draft path must be safe and relative: {destination}")
    return path.parts


def _build_draft(build: Callable[[Draft], None]) -> Callable[[Path, int, int], None]:
    def run(path: Path, descriptor: int, device: int) -> None:
        build(Draft(path, descriptor, device))

    return run


def publish(destination: Path, build: Callable[[Draft], None]) -> Publication:
    committed = _posix.publication_attempt(destination, _build_draft(build))
    return Publication(committed, cleanup_pending=False)


class Workspace:
    def __init__(self, destination: Path, *, identity: bytes) -> None:
        parent, destination_name = _posix.normalize_destination(destination)
        destination_key = hashlib.sha256(os.fsencode(destination_name)).hexdigest()[:24]
        self._destination = parent / destination_name
        self._parent = parent
        self._parent_fd = -1
        self._container_fd = -1
        self._work_fd = -1
        self._lock_fd = -1
        self._container_entry: os.stat_result | None = None
        self._container_name = f".servatus-{destination_key}.work"
        self._identity = hashlib.sha256(identity).hexdigest().encode("ascii") + b"\n"
        self._entered = False
        self._published = False

    def __enter__(self) -> Workspace:
        if self._entered:
            raise RuntimeError("workspace is already entered")
        _posix.require_supported_platform()
        self._parent_fd = _posix.open_directory(self._parent)
        try:
            self._container_fd, self._container_entry = _posix.make_directory_at(
                self._parent_fd, self._container_name
            )
            self._lock_fd = self._open_lock()
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise WorkspaceBusy(f"workspace is already locked: {self._destination}") from error
            self._bind_identity()
            self._work_fd, _ = _posix.make_directory_at(self._container_fd, "work")
            _posix.sync_descriptor(self._container_fd)
        except BaseException:
            self._close()
            raise
        self._entered = True
        return self

    def __exit__(self, *_: object) -> None:
        self._close()

    @property
    def path(self) -> Path:
        return self._parent / self._container_name / "work"

    def publish(self, build: Callable[[Draft], None]) -> Publication:
        if not self._entered or self._parent_fd < 0 or self._container_entry is None:
            raise RuntimeError("workspace must be entered before publication")
        if self._published:
            raise RuntimeError("workspace has already published")
        _posix.ensure_entry(self._parent_fd, self._container_name, self._container_entry)
        self._verify_identity()
        committed = _posix.publication_attempt_at(
            self._parent,
            self._parent_fd,
            self._destination.name,
            _build_draft(build),
        )
        cleanup_pending = False
        try:
            _cleanup_workspace(
                self._parent_fd,
                self._container_name,
                self._container_entry,
            )
        except Exception:
            cleanup_pending = True
        self._published = True
        return Publication(committed, cleanup_pending=cleanup_pending)

    def _open_lock(self) -> int:
        assert self._container_fd >= 0
        try:
            descriptor = os.open(
                ".lock",
                os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=self._container_fd,
            )
        except OSError as error:
            raise UnsafePublication("workspace lock is unsafe") from error
        lock_entry = os.fstat(descriptor)
        if not stat.S_ISREG(lock_entry.st_mode):
            os.close(descriptor)
            raise UnsafePublication("workspace lock is not a regular file")
        return descriptor

    def _bind_identity(self) -> None:
        assert self._container_fd >= 0
        try:
            descriptor = os.open(
                ".identity",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=self._container_fd,
            )
        except FileExistsError:
            self._verify_identity()
            return
        try:
            view = memoryview(self._identity)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            _posix.sync_descriptor(descriptor)
        finally:
            os.close(descriptor)
        _posix.sync_descriptor(self._container_fd)

    def _verify_identity(self) -> None:
        assert self._container_fd >= 0
        try:
            descriptor = os.open(
                ".identity",
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=self._container_fd,
            )
        except OSError as error:
            raise WorkConflict(f"workspace identity is unavailable: {self._destination}") from error
        try:
            identity_entry = os.fstat(descriptor)
            if not stat.S_ISREG(identity_entry.st_mode):
                raise WorkConflict(f"workspace identity is invalid: {self._destination}")
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 4096):
                chunks.append(chunk)
            if b"".join(chunks) != self._identity:
                raise WorkConflict(f"workspace belongs to different work: {self._destination}")
        finally:
            os.close(descriptor)

    def _close(self) -> None:
        for attribute in ("_work_fd", "_lock_fd", "_container_fd", "_parent_fd"):
            descriptor = getattr(self, attribute)
            if descriptor >= 0:
                with suppress(OSError):
                    os.close(descriptor)
                setattr(self, attribute, -1)
        self._entered = False


def _cleanup_workspace(
    parent_fd: int,
    name: str,
    expected: os.stat_result,
) -> None:
    _posix.remove_tree_at(parent_fd, name, expected)
