from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import stat
import warnings
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from . import _posix
from ._errors import (
    CrossDevicePublication,
    DestinationExists,
    UnsafePublication,
    WorkConflict,
    WorkspaceBusy,
)

_WORKSPACE_STATE_HEADER = b"servatus-workspace-v2\n"
_MAX_INODE = (1 << 64) - 1
_MAX_WORKSPACE_STATE_BYTES = len(_WORKSPACE_STATE_HEADER) + 65 + 3 * (len(str(_MAX_INODE)) + 1)


@dataclass(frozen=True, slots=True)
class Publication:
    destination: Path
    cleanup_pending: bool


def _publication_result(destination: Path, *, cleanup_pending: bool) -> Publication:
    publication = Publication(destination, cleanup_pending=cleanup_pending)
    if cleanup_pending:
        _warn_nonfatal("publication committed, but private cleanup remains pending", stacklevel=4)
    return publication


def _warn_nonfatal(message: str, *, stacklevel: int) -> None:
    with suppress(BaseException):
        warnings.warn(message, RuntimeWarning, stacklevel=stacklevel)


@dataclass(slots=True)
class _WorkspaceLevel:
    container_fd: int = -1
    lock_fd: int = -1
    work_fd: int = -1
    container_entry: os.stat_result | None = None
    lock_entry: os.stat_result | None = None
    work_entry: os.stat_result | None = None


class Draft:
    def __init__(self, path: Path, descriptor: int) -> None:
        self._path = path
        self._descriptor = descriptor
        self._failure: UnsafePublication | None = None

    @property
    def path(self) -> Path:
        return self._path

    def link(self, source: Path, destination: str | PurePosixPath) -> None:
        components = _safe_components(destination)
        source_path = Path(source)
        _posix.reject_nul_path(source_path)
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
                        f"hard-link source is on another filesystem: {source_path}"
                    ) from error
                raise UnsafePublication(f"unsafe hard-link source: {source_path}") from error
            try:
                linked = os.stat(leaf, dir_fd=current_fd, follow_symlinks=False)
            except OSError as error:
                try:
                    os.unlink(leaf, dir_fd=current_fd)
                except OSError as cleanup_error:
                    cleanup_error.add_note(f"Inspection failed: {error}")
                    failure = UnsafePublication(
                        f"hard-link destination could not be inspected or removed: {destination}"
                    )
                    self._failure = failure
                    raise failure from cleanup_error
                raise UnsafePublication(
                    f"hard-link destination is unavailable: {destination}"
                ) from error
            if stat.S_ISREG(linked.st_mode):
                return
            try:
                os.unlink(leaf, dir_fd=current_fd)
            except OSError as error:
                failure = UnsafePublication(
                    f"unsafe hard-link source could not be removed: {source_path}"
                )
                self._failure = failure
                raise failure from error
            raise UnsafePublication(f"hard-link source is not a regular file: {source_path}")
        finally:
            os.close(current_fd)


def _safe_components(destination: str | PurePosixPath) -> tuple[str, ...]:
    if "\0" in os.fspath(destination):
        raise UnsafePublication(f"draft path contains an embedded NUL: {destination!r}")
    path = PurePosixPath(destination)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise UnsafePublication(f"draft path must be safe and relative: {destination}")
    return path.parts


def _build_draft(build: Callable[[Draft], None]) -> Callable[[Path, int], None]:
    def run(path: Path, descriptor: int) -> None:
        draft = Draft(path, descriptor)
        build(draft)
        if failure := draft._failure:  # pyright: ignore[reportPrivateUsage]
            raise failure

    return run


def publish(
    destination: Path,
    build: Callable[[Draft], None],
    *,
    retire: Path | None = None,
) -> Publication:
    outcome = _posix.publication_attempt(destination, _build_draft(build), retire=retire)
    return _publication_result(outcome.destination, cleanup_pending=outcome.cleanup_pending)


def publish_file(destination: Path, write: Callable[[Path], None]) -> Publication:
    outcome = _posix.file_publication_attempt(destination, write)
    return _publication_result(outcome.destination, cleanup_pending=outcome.cleanup_pending)


@dataclass(frozen=True, slots=True)
class _WorkspaceLocation:
    destination: Path
    identity: bytes
    root: _WorkspaceLocation | None = None

    @property
    def parent(self) -> Path:
        return self.destination.parent

    @property
    def container_name(self) -> str:
        return _container_name(self.destination.name)

    @property
    def path(self) -> Path:
        return self.parent / self.container_name / "work"


@dataclass(slots=True)
class _WorkspaceSession:
    parent_fd: int = -1
    root_level: _WorkspaceLevel | None = None
    level: _WorkspaceLevel = field(default_factory=_WorkspaceLevel)
    entered: bool = False
    published: bool = False


class Workspace:
    def __init__(self, destination: Path, *, identity: bytes) -> None:
        parent, destination_name = _posix.normalize_destination(destination)
        self._location = _WorkspaceLocation(parent / destination_name, _identity_digest(identity))
        self._session = _WorkspaceSession()

    def child(self, name: str, *, identity: bytes) -> Workspace:
        if self._location.root is not None:
            raise UnsafePublication("child workspaces cannot contain child workspaces")
        _posix.validate_leaf(name)
        child = Workspace(self._location.destination, identity=identity)
        child._location = _WorkspaceLocation(
            self.path / name, child._location.identity, self._location
        )
        return child

    def __enter__(self) -> Workspace:
        if self._session.entered:
            raise RuntimeError("workspace is already entered")
        _posix.require_supported_platform()
        location = self._location
        root = location.root or location
        session = self._session = _WorkspaceSession()
        session.parent_fd = _posix.open_directory(root.parent)
        try:
            _posix.ensure_directory_path(root.parent, session.parent_fd)
            _posix.ensure_absent(session.parent_fd, root.destination.name, root.destination)
            if location.root is None:
                session.level = _open_level(
                    session.parent_fd,
                    location.container_name,
                    location.identity,
                    location.destination,
                    fcntl.LOCK_EX,
                )
            else:
                session.root_level = _open_level(
                    session.parent_fd,
                    root.container_name,
                    root.identity,
                    root.destination,
                    fcntl.LOCK_SH,
                )
                session.level = _open_level(
                    session.root_level.work_fd,
                    location.container_name,
                    location.identity,
                    location.destination,
                    fcntl.LOCK_EX,
                )
            self._verify_live()
            _posix.ensure_absent(session.parent_fd, root.destination.name, root.destination)
            if session.root_level is not None:
                _posix.ensure_absent(
                    session.root_level.work_fd, location.destination.name, location.destination
                )
        except BaseException:
            self._close()
            raise
        session.entered = True
        return self

    def __exit__(self, *_: object) -> None:
        self._close()

    @property
    def path(self) -> Path:
        return self._location.path

    def publish(self, build: Callable[[Draft], None]) -> Publication:
        session = self._session
        location = self._location
        if not session.entered or session.level.container_entry is None:
            raise RuntimeError("workspace must be entered before publication")
        if session.published:
            raise RuntimeError("workspace has already published")
        self._verify_live()
        outcome = _posix.publication_attempt_at(
            location.parent,
            self._publication_parent_fd(),
            location.destination.name,
            _build_draft(build),
        )
        cleanup_pending = outcome.cleanup_pending
        try:
            self._cleanup_published()
        except Exception:
            cleanup_pending = True
        session.published = True
        return _publication_result(outcome.destination, cleanup_pending=cleanup_pending)

    def _publication_parent_fd(self) -> int:
        session = self._session
        if session.root_level is None:
            return session.parent_fd
        return session.root_level.work_fd

    def _verify_live(self) -> None:
        location = self._location
        session = self._session
        root = location.root or location
        _posix.ensure_directory_path(root.parent, session.parent_fd)
        if session.root_level is not None:
            _verify_level(session.parent_fd, root.container_name, session.root_level)
            _verify_identity(
                session.root_level.container_fd, root.identity, root.destination, session.root_level
            )
        _verify_level(self._publication_parent_fd(), location.container_name, session.level)
        _verify_identity(
            session.level.container_fd, location.identity, location.destination, session.level
        )

    def _cleanup_published(self) -> None:
        assert self._session.level.container_entry is not None
        self._verify_live()
        _posix.remove_tree_at(
            self._publication_parent_fd(),
            self._location.container_name,
            self._session.level.container_entry,
        )

    def _close(self) -> None:
        session = self._session
        _close_level(session.level)
        if session.root_level is not None:
            _close_level(session.root_level)
        session.entered = False
        if session.parent_fd >= 0:
            with suppress(OSError):
                os.close(session.parent_fd)
            session.parent_fd = -1


def _container_name(destination_name: str) -> str:
    destination_key = hashlib.sha256(os.fsencode(destination_name)).hexdigest()[:24]
    return f".servatus-{destination_key}.work"


def _identity_digest(identity: bytes) -> bytes:
    return hashlib.sha256(identity).hexdigest().encode("ascii") + b"\n"


@contextmanager
def _coordinate(parent_fd: int) -> Generator[None]:
    fcntl.flock(parent_fd, fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(parent_fd, fcntl.LOCK_UN)


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
        _posix.require_owner_only(lock_entry, "workspace lock .lock")
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
        with _coordinate(parent_fd):
            level.container_fd, level.container_entry = _posix.make_directory_at(
                parent_fd, container_name
            )
            _posix.require_owner_only(
                level.container_entry, f"workspace container {container_name}"
            )
            level.lock_fd, level.lock_entry = _open_lock(level.container_fd)
            _acquire_lifecycle(level.lock_fd, lock_mode, destination)
            _posix.ensure_absent(parent_fd, destination.name, destination)
            level.work_fd, level.work_entry = _posix.make_directory_at(level.container_fd, "work")
        _verify_level(parent_fd, container_name, level)
        _bind_identity(parent_fd, level.container_fd, identity, destination, level)
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
    parent_fd: int,
    container_fd: int,
    identity: bytes,
    destination: Path,
    level: _WorkspaceLevel,
) -> None:
    try:
        os.stat(".identity", dir_fd=container_fd, follow_symlinks=False)
    except FileNotFoundError:
        _sync_workspace_initialization(parent_fd, level)
        _initialize_identity(container_fd, identity, destination, level)
    else:
        _verify_identity(container_fd, identity, destination, level)


def _sync_workspace_initialization(parent_fd: int, level: _WorkspaceLevel) -> None:
    _posix.sync_descriptor(level.lock_fd)
    _posix.sync_descriptor(level.work_fd)
    _posix.sync_descriptor(level.container_fd)
    _posix.sync_descriptor(parent_fd)


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
            commit = _posix.commit_noreplace(container_fd, stage_name, ".identity", stage_entry)
            if commit.cleanup_pending:
                _warn_nonfatal(
                    "workspace identity was installed, but identity-stage cleanup remains pending",
                    stacklevel=4,
                )
        except DestinationExists:
            _verify_identity(container_fd, identity, destination, level)
            _posix.remove_file_at(container_fd, stage_name, stage_entry)
            stage_entry = None
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
        _posix.require_owner_only(identity_entry, "workspace identity .identity")
        chunks: list[bytes] = []
        remaining = identity_entry.st_size + 1
        while remaining and (chunk := os.read(descriptor, remaining)):
            chunks.append(chunk)
            remaining -= len(chunk)
        actual = b"".join(chunks)
        identity_prefix = _WORKSPACE_STATE_HEADER + identity
        if not actual.startswith(identity_prefix):
            raise WorkConflict(f"workspace belongs to different work: {destination}")
        try:
            persisted_inodes = _parse_workspace_pins(actual[len(identity_prefix) :])
        except ValueError:
            raise UnsafePublication(
                f"workspace lifecycle record is invalid: {destination}"
            ) from None
        assert level.container_entry is not None
        assert level.lock_entry is not None
        assert level.work_entry is not None
        current_inodes = (
            level.container_entry.st_ino,
            level.lock_entry.st_ino,
            level.work_entry.st_ino,
        )
        if persisted_inodes != current_inodes:
            raise UnsafePublication(f"workspace lifecycle changed: {destination}")
        _posix.ensure_entry(container_fd, ".identity", identity_entry)
    finally:
        os.close(descriptor)


def _ensure_open_entry(
    parent_fd: int,
    name: str,
    descriptor: int,
    expected: os.stat_result,
) -> None:
    _posix.require_owner_only(os.fstat(descriptor), f"workspace entry {name}")
    _posix.ensure_entry(parent_fd, name, expected)


def _workspace_state(identity: bytes, level: _WorkspaceLevel) -> bytes:
    assert level.container_entry is not None
    assert level.lock_entry is not None
    assert level.work_entry is not None
    entries = (level.container_entry, level.lock_entry, level.work_entry)
    pins = b"".join(f"{entry.st_ino}\n".encode("ascii") for entry in entries)
    return _WORKSPACE_STATE_HEADER + identity + pins


def _parse_workspace_pins(data: bytes) -> tuple[int, int, int]:
    lines = data.split(b"\n")
    if len(lines) != 4 or lines[-1]:
        raise ValueError("workspace lifecycle pin count is invalid")
    inodes: list[int] = []
    for line in lines[:-1]:
        if not line.isdigit():
            raise ValueError("workspace lifecycle pin is invalid")
        inode = int(line)
        if inode > _MAX_INODE or line != str(inode).encode("ascii"):
            raise ValueError("workspace lifecycle pin is not canonical")
        inodes.append(inode)
    return inodes[0], inodes[1], inodes[2]


def _verify_level(parent_fd: int, container_name: str, level: _WorkspaceLevel) -> None:
    assert level.container_entry is not None
    assert level.lock_entry is not None
    assert level.work_entry is not None
    _ensure_open_entry(
        parent_fd,
        container_name,
        level.container_fd,
        level.container_entry,
    )
    _ensure_open_entry(
        level.container_fd,
        ".lock",
        level.lock_fd,
        level.lock_entry,
    )
    _ensure_open_entry(
        level.container_fd,
        "work",
        level.work_fd,
        level.work_entry,
    )


def _close_level(level: _WorkspaceLevel) -> None:
    for attribute in ("work_fd", "lock_fd", "container_fd"):
        descriptor = getattr(level, attribute)
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)
            setattr(level, attribute, -1)
