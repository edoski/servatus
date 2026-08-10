from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import stat
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
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

_WORKSPACE_STATE_HEADER = b"servatus-workspace-v1\n"
_MAX_WORKSPACE_STATE_BYTES = 512


@dataclass(frozen=True, slots=True)
class Publication:
    destination: Path
    cleanup_pending: bool


@dataclass(slots=True)
class _WorkspaceLevel:
    container_fd: int = -1
    lock_fd: int = -1
    work_fd: int = -1
    container_entry: os.stat_result | None = None
    lock_entry: os.stat_result | None = None
    work_entry: os.stat_result | None = None


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
        _posix.reject_nul_path(source_path)
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
    if "\0" in os.fspath(destination):
        raise UnsafePublication(f"draft path contains an embedded NUL: {destination!r}")
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


def publish_file(destination: Path, write: Callable[[Path], None]) -> Publication:
    committed = _posix.file_publication_attempt(destination, write)
    return Publication(committed, cleanup_pending=False)


class Workspace:
    def __init__(self, destination: Path, *, identity: bytes) -> None:
        parent, destination_name = _posix.normalize_destination(destination)
        self._root_parent = parent
        self._root_destination_name = destination_name
        self._root_container_name = _container_name(destination_name)
        self._root_identity = _identity_digest(identity)
        self._child_name: str | None = None
        self._destination = parent / destination_name
        self._parent = parent
        self._container_name = self._root_container_name
        self._identity = self._root_identity
        self._initialize_state()

    def _initialize_state(self) -> None:
        self._stable_parent_fd = -1
        self._root_level: _WorkspaceLevel | None = None
        self._level = _WorkspaceLevel()
        self._entered = False
        self._published = False

    def child(self, name: str, *, identity: bytes) -> Workspace:
        if self._child_name is not None:
            raise UnsafePublication("child workspaces cannot contain child workspaces")
        _posix.validate_leaf(name)
        child = object.__new__(Workspace)
        child._root_parent = self._root_parent
        child._root_destination_name = self._root_destination_name
        child._root_container_name = self._root_container_name
        child._root_identity = self._root_identity
        child._child_name = name
        child._parent = self.path
        child._destination = child._parent / name
        child._container_name = _container_name(name)
        child._identity = _identity_digest(identity)
        child._initialize_state()
        return child

    def __enter__(self) -> Workspace:
        if self._entered:
            raise RuntimeError("workspace is already entered")
        _posix.require_supported_platform()
        self._stable_parent_fd = _posix.open_directory(self._root_parent)
        try:
            with _coordinate(self._stable_parent_fd):
                _posix.ensure_directory_path(self._root_parent, self._stable_parent_fd)
                _ensure_absent(
                    self._stable_parent_fd,
                    self._root_destination_name,
                    self._root_parent / self._root_destination_name,
                )
                if self._child_name is None:
                    self._enter_root()
                else:
                    self._enter_child()
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
        if not self._entered or self._level.container_entry is None:
            raise RuntimeError("workspace must be entered before publication")
        if self._published:
            raise RuntimeError("workspace has already published")
        self._verify_live()
        publication_parent_fd = self._publication_parent_fd()
        committed = _posix.publication_attempt_at(
            self._parent,
            publication_parent_fd,
            self._destination.name,
            _build_draft(build),
        )
        cleanup_pending = False
        try:
            self._cleanup_published()
        except Exception:
            cleanup_pending = True
        self._published = True
        return Publication(committed, cleanup_pending=cleanup_pending)

    def _enter_root(self) -> None:
        self._level = _open_level(
            self._stable_parent_fd,
            self._root_container_name,
            self._identity,
            self._destination,
            fcntl.LOCK_EX,
        )
        self._verify_live()

    def _enter_child(self) -> None:
        assert self._child_name is not None
        root_destination = self._root_parent / self._root_destination_name
        self._root_level = _open_level(
            self._stable_parent_fd,
            self._root_container_name,
            self._root_identity,
            root_destination,
            fcntl.LOCK_SH,
        )
        _ensure_absent(self._root_level.work_fd, self._child_name, self._destination)
        self._level = _open_level(
            self._root_level.work_fd,
            self._container_name,
            self._identity,
            self._destination,
            fcntl.LOCK_EX,
        )
        self._verify_live()

    def _publication_parent_fd(self) -> int:
        if self._child_name is None:
            return self._stable_parent_fd
        assert self._root_level is not None
        return self._root_level.work_fd

    def _verify_live(self) -> None:
        _posix.ensure_directory_path(self._root_parent, self._stable_parent_fd)
        if self._child_name is None:
            _verify_level(self._stable_parent_fd, self._root_container_name, self._level)
        else:
            assert self._root_level is not None
            _verify_level(self._stable_parent_fd, self._root_container_name, self._root_level)
            _verify_identity(
                self._root_level.container_fd,
                self._root_identity,
                self._root_parent / self._root_destination_name,
                self._root_level,
            )
            _verify_level(self._root_level.work_fd, self._container_name, self._level)
        _verify_identity(
            self._level.container_fd,
            self._identity,
            self._destination,
            self._level,
        )

    def _cleanup_published(self) -> None:
        assert self._level.container_entry is not None
        with _coordinate(self._stable_parent_fd):
            self._verify_live()
            _cleanup_workspace(
                self._publication_parent_fd(),
                self._container_name,
                self._level.container_entry,
            )

    def _close(self) -> None:
        _close_level(self._level)
        if self._root_level is not None:
            _close_level(self._root_level)
        self._entered = False
        if self._stable_parent_fd >= 0:
            with suppress(OSError):
                os.close(self._stable_parent_fd)
            self._stable_parent_fd = -1


def _cleanup_workspace(
    parent_fd: int,
    name: str,
    expected: os.stat_result,
) -> None:
    _posix.remove_tree_at(parent_fd, name, expected)


def _container_name(destination_name: str) -> str:
    destination_key = hashlib.sha256(os.fsencode(destination_name)).hexdigest()[:24]
    return f".servatus-{destination_key}.work"


def _identity_digest(identity: bytes) -> bytes:
    return hashlib.sha256(identity).hexdigest().encode("ascii") + b"\n"


@contextmanager
def _coordinate(stable_parent_fd: int) -> Generator[None]:
    fcntl.flock(stable_parent_fd, fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(stable_parent_fd, fcntl.LOCK_UN)


def _ensure_absent(parent_fd: int, name: str, destination: Path) -> None:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as error:
        raise UnsafePublication(f"publication destination is unavailable: {destination}") from error
    raise DestinationExists(f"publication destination already exists: {destination}")


def _open_lock(container_fd: int) -> tuple[int, os.stat_result]:
    try:
        descriptor = os.open(
            ".lock",
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=container_fd,
        )
    except OSError as error:
        raise UnsafePublication("workspace lock is unsafe") from error
    lock_entry = os.fstat(descriptor)
    if not stat.S_ISREG(lock_entry.st_mode) or lock_entry.st_dev != os.fstat(container_fd).st_dev:
        os.close(descriptor)
        raise UnsafePublication("workspace lock is not a regular file")
    try:
        _posix.ensure_entry(container_fd, ".lock", lock_entry)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, lock_entry


def _open_level(
    parent_fd: int,
    container_name: str,
    identity: bytes,
    destination: Path,
    lock_mode: int,
) -> _WorkspaceLevel:
    level = _WorkspaceLevel()
    try:
        level.container_fd, level.container_entry = _posix.make_directory_at(
            parent_fd, container_name
        )
        level.lock_fd, level.lock_entry = _open_lock(level.container_fd)
        _acquire_lifecycle(level.lock_fd, lock_mode, destination)
        level.work_fd, level.work_entry = _posix.make_directory_at(level.container_fd, "work")
        _verify_level(parent_fd, container_name, level)
        _bind_identity(level.container_fd, identity, destination, level)
        _posix.sync_descriptor(level.container_fd)
        _verify_level(parent_fd, container_name, level)
    except BaseException:
        _close_level(level)
        raise
    return level


def _acquire_lifecycle(descriptor: int, mode: int, destination: Path) -> None:
    try:
        fcntl.flock(descriptor, mode | fcntl.LOCK_NB)
    except OSError as error:
        if error.errno in {errno.EACCES, errno.EAGAIN}:
            raise WorkspaceBusy(f"workspace is already locked: {destination}") from error
        raise


def _bind_identity(
    container_fd: int,
    identity: bytes,
    destination: Path,
    level: _WorkspaceLevel,
) -> None:
    try:
        os.stat(".identity", dir_fd=container_fd, follow_symlinks=False)
    except FileNotFoundError:
        _initialize_identity(container_fd, identity, destination, level)
    else:
        _verify_identity(container_fd, identity, destination, level)


def _initialize_identity(
    container_fd: int,
    identity: bytes,
    destination: Path,
    level: _WorkspaceLevel,
) -> None:
    state = _workspace_state(identity, level)
    stage_name = f".identity-{os.urandom(12).hex()}.tmp"
    descriptor = -1
    stage_entry: os.stat_result | None = None
    try:
        descriptor = os.open(
            stage_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=container_fd,
        )
        stage_entry = os.fstat(descriptor)
        view = memoryview(state)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        _posix.sync_descriptor(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            _posix.commit_noreplace(container_fd, stage_name, ".identity")
        except DestinationExists:
            _verify_identity(container_fd, identity, destination, level)
            _posix.remove_file_at(container_fd, stage_name, stage_entry)
            stage_entry = None
        _posix.sync_descriptor(container_fd)
    except BaseException as error:
        if stage_entry is not None:
            try:
                _posix.remove_file_at(container_fd, stage_name, stage_entry)
            except Exception as cleanup_error:
                error.add_note(f"Servatus could not remove identity stage: {cleanup_error}")
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _verify_identity(
    container_fd: int,
    identity: bytes,
    destination: Path,
    level: _WorkspaceLevel,
) -> None:
    try:
        descriptor = os.open(
            ".identity",
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=container_fd,
        )
    except OSError as error:
        raise WorkConflict(f"workspace identity is unavailable: {destination}") from error
    try:
        identity_entry = os.fstat(descriptor)
        if (
            not stat.S_ISREG(identity_entry.st_mode)
            or identity_entry.st_dev != os.fstat(container_fd).st_dev
            or identity_entry.st_size > _MAX_WORKSPACE_STATE_BYTES
        ):
            raise WorkConflict(f"workspace identity is invalid: {destination}")
        chunks: list[bytes] = []
        remaining = identity_entry.st_size + 1
        while remaining and (chunk := os.read(descriptor, remaining)):
            chunks.append(chunk)
            remaining -= len(chunk)
        actual = b"".join(chunks)
        identity_prefix = _WORKSPACE_STATE_HEADER + identity
        if not actual.startswith(identity_prefix):
            raise WorkConflict(f"workspace belongs to different work: {destination}")
        if actual != _workspace_state(identity, level):
            raise UnsafePublication(f"workspace lifecycle changed: {destination}")
        _posix.ensure_entry(container_fd, ".identity", identity_entry)
    finally:
        os.close(descriptor)


def _ensure_open_entry(
    parent_fd: int,
    name: str,
    descriptor: int,
    expected: os.stat_result,
    *,
    directory: bool,
) -> None:
    opened = os.fstat(descriptor)
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_type(opened.st_mode) or not _posix.same_entry(opened, expected):
        raise UnsafePublication(f"workspace entry changed: {name}")
    _posix.ensure_entry(parent_fd, name, expected)


def _workspace_state(identity: bytes, level: _WorkspaceLevel) -> bytes:
    assert level.container_entry is not None
    assert level.lock_entry is not None
    assert level.work_entry is not None
    entries = (level.container_entry, level.lock_entry, level.work_entry)
    pins = b"".join(f"{entry.st_dev}:{entry.st_ino}\n".encode("ascii") for entry in entries)
    return _WORKSPACE_STATE_HEADER + identity + pins


def _verify_level(parent_fd: int, container_name: str, level: _WorkspaceLevel) -> None:
    assert level.container_entry is not None
    assert level.lock_entry is not None
    assert level.work_entry is not None
    _ensure_open_entry(
        parent_fd,
        container_name,
        level.container_fd,
        level.container_entry,
        directory=True,
    )
    _ensure_open_entry(
        level.container_fd,
        ".lock",
        level.lock_fd,
        level.lock_entry,
        directory=False,
    )
    _ensure_open_entry(
        level.container_fd,
        "work",
        level.work_fd,
        level.work_entry,
        directory=True,
    )


def _close_level(level: _WorkspaceLevel) -> None:
    for attribute in ("work_fd", "lock_fd", "container_fd"):
        descriptor = getattr(level, attribute)
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)
            setattr(level, attribute, -1)
