"""Pinned-descriptor POSIX primitives shared by publication and the campaign store.

Nothing here knows about publication policy. Every descriptor is wrapped in a `Pin` the moment it
is opened: the pin is verified (device, inode, type) or the descriptor is closed before the error
propagates, so no failure path leaks a descriptor. Names are always resolved relative to a pinned
directory and never followed.

Public operations run inside a `Boundary`, the one place where an `OSError` becomes the public
error a caller can act on (`os_error`); `UnsafeFilesystem` is reserved for the explicit
substitution, type, and ownership checks made here.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import platform
import stat
import sys
from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager, suppress
from functools import cache
from pathlib import Path
from types import TracebackType
from typing import Any, ParamSpec, TypeAlias, TypeVar

from .errors import (
    ConfigurationError,
    CrossDeviceError,
    DestinationExists,
    ServatusError,
    Unavailable,
    UnsafeFilesystem,
    UnsupportedPlatform,
)

P = ParamSpec("P")
R = TypeVar("R")
StrPath: TypeAlias = "str | os.PathLike[str]"
NoReplace: TypeAlias = Callable[[int, bytes, int, bytes], int]
"""A native no-replace rename returning 0 or an errno value."""

LOCK_NAME = ".servatus.lock"
"""The owner-only lock file used where a directory handle cannot be `flock`ed (NFS)."""

_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
_READ = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_NEW = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
_LOCK = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_CHUNK = 1024 * 1024
_UNSUPPORTED = frozenset({errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP})
_NO_DIRECTORY_LOCK = frozenset(
    {errno.EBADF, errno.ENOLCK, errno.EOPNOTSUPP, errno.ENOTSUP, errno.EINVAL}
)
_BAD_LOCATION = frozenset(
    {
        errno.EACCES,
        errno.EPERM,
        errno.EROFS,
        errno.ENAMETOOLONG,
        errno.ENOTDIR,
        errno.EISDIR,
        errno.ELOOP,
        errno.EEXIST,
    }
)
_NOT_PLAIN = frozenset({errno.ELOOP, errno.ENOTDIR, errno.EISDIR, errno.ENXIO, *_UNSUPPORTED})
_DENIED = frozenset({errno.EACCES, errno.EPERM})
_CHMOD_NOFOLLOW = os.chmod in os.supports_follow_symlinks
_RENAME_NOREPLACE = 0x1  # Linux renameat2 flag
_RENAME_EXCL = 0x4  # macOS renameatx_np flag
_SYS_RENAMEAT2 = {"x86_64": 316, "amd64": 316, "aarch64": 276, "arm64": 276}


# -- errors ----------------------------------------------------------------------------------
def os_error(
    error: OSError, subject: object, *, missing: type[ServatusError] = ConfigurationError
) -> ServatusError:
    """Map one filesystem `OSError` about `subject` to the public error a caller can act on.

    A missing entry is `missing`; denied access, read-only filesystems, overlong names, and paths
    of the wrong kind are a bad location (`ConfigurationError`); a cross-device operation is
    `CrossDeviceError`; everything else (no space, quota, I/O errors, descriptor exhaustion) is
    transient: `Unavailable`, retry later.
    """
    code = error.errno
    if code == errno.ENOENT:
        return missing(f"{subject} does not exist: {error}")
    if code in _BAD_LOCATION:
        return ConfigurationError(f"{subject} is not a usable location: {error}")
    if code == errno.EXDEV:
        return CrossDeviceError(f"{subject} crosses a filesystem boundary: {error}")
    return Unavailable(f"filesystem operation on {subject} failed; retry later: {error}")


class Boundary:
    """Map every `OSError` escaping a public operation through `os_error`.

    Exceptions raised by caller callbacks wrapped with `passthrough` keep their identity.
    """

    __slots__ = ("_foreign", "_missing", "_subject")

    def __init__(self, subject: object, *, missing: type[ServatusError] = ConfigurationError):
        self._subject = subject
        self._missing = missing
        self._foreign: BaseException | None = None

    def passthrough(self, function: Callable[P, R]) -> Callable[P, R]:
        def call(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return function(*args, **kwargs)
            except OSError as error:
                self._foreign = error
                raise

        return call

    def __enter__(self) -> Boundary:
        return self

    def __exit__(
        self,
        _kind: type[BaseException] | None,
        error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        if isinstance(error, OSError) and error is not self._foreign:
            mapped = os_error(error, self._subject, missing=self._missing)
            for note in getattr(error, "__notes__", ()):
                mapped.add_note(note)
            raise mapped from error


def succeeded(function: Callable[P, object], *args: P.args, **kwargs: P.kwargs) -> bool:
    """Run one best-effort step and report whether it completed. Interrupts still propagate."""
    try:
        function(*args, **kwargs)
    except Exception:
        return False
    return True


# -- validation ------------------------------------------------------------------------------
def require_supported_platform() -> None:
    if not (sys.platform.startswith("linux") or sys.platform == "darwin"):
        raise UnsupportedPlatform(f"publication requires Linux or macOS, not {sys.platform}")


def fspath(path: StrPath) -> str:
    """Return a text path, rejecting bytes paths and embedded NULs."""
    text = os.fspath(path)
    if not isinstance(text, str):
        raise ConfigurationError(f"path must be text: {path!r}")
    if "\0" in text:
        raise ConfigurationError(f"path contains an embedded NUL: {text!r}")
    return text


def leaf(name: str) -> str:
    """Return `name` if it is one directory entry name; never a separator, `.`, or `..`."""
    if not isinstance(name, str) or name in {"", ".", ".."} or "/" in name or "\0" in name:
        raise ConfigurationError(f"unsafe filesystem leaf: {name!r}")
    return name


def check_mode(mode: int) -> int:
    if isinstance(mode, bool) or not isinstance(mode, int) or not 0 <= mode <= 0o777:
        raise ConfigurationError(f"mode must be a permission set from 0 to 0o777: {mode!r}")
    return mode


def same(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def require_owner_only(entry: os.stat_result, what: str) -> None:
    if entry.st_uid != os.geteuid() or stat.S_IMODE(entry.st_mode) & 0o077:
        raise UnsafeFilesystem(f"{what} must be owner-only and owned by the current user")


def split(destination: StrPath) -> tuple[Path, str]:
    """Resolve the parent once (symlinks followed); the final name is never followed."""
    raw = Path(fspath(destination))
    if raw.name in {"", ".", ".."}:
        raise ConfigurationError(f"destination must name an entry in an existing parent: {raw}")
    try:
        return raw.parent.resolve(strict=True), raw.name
    except (OSError, RuntimeError) as error:
        raise ConfigurationError(f"destination parent is unavailable: {raw.parent}") from error


# -- durability ------------------------------------------------------------------------------
def sync(fd: int) -> None:
    """Flush one descriptor to stable storage (`F_FULLFSYNC` on macOS, where fsync does not)."""
    if sys.platform == "darwin":
        try:
            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
        except OSError:
            pass  # some filesystems reject F_FULLFSYNC; fsync is the best remaining request
        else:
            return
    os.fsync(fd)


# -- pins ------------------------------------------------------------------------------------
class Pin:
    """One open descriptor and the inode it was verified as. Holding it prevents inode reuse."""

    __slots__ = ("_entry", "_fd")

    def __init__(self, fd: int, entry: os.stat_result) -> None:
        self._fd = fd
        self._entry = entry

    @property
    def fd(self) -> int:
        if self._fd < 0:
            raise RuntimeError("pinned descriptor is already closed")
        return self._fd

    @property
    def entry(self) -> os.stat_result:
        return self._entry

    def close(self) -> None:
        """Close once; later calls do nothing, so a reused descriptor number is never closed."""
        fd, self._fd = self._fd, -1
        if fd >= 0:
            with suppress(OSError):
                os.close(fd)

    def __enter__(self) -> Pin:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # Name/inode checks relative to this directory.
    def lstat(self, name: str) -> os.stat_result | None:
        try:
            return os.stat(leaf(name), dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError:
            return None

    def present(self, name: str) -> os.stat_result:
        entry = self.lstat(name)
        if entry is None:
            raise UnsafeFilesystem(f"filesystem entry disappeared: {name}")
        return entry

    def expect(self, name: str, expected: os.stat_result) -> os.stat_result:
        entry = self.present(name)
        if not same(entry, expected):
            raise UnsafeFilesystem(f"filesystem entry was substituted: {name}")
        return entry

    def absent(self, name: str, shown: object) -> None:
        if self.lstat(name) is not None:
            raise DestinationExists(f"destination already exists: {shown}")

    def verify_path(self, path: Path) -> None:
        """Require that `path` still names this pinned directory."""
        try:
            current = os.stat(path, follow_symlinks=False)
        except OSError as error:
            raise UnsafeFilesystem(f"directory disappeared: {path}") from error
        if not same(current, self.entry):
            raise UnsafeFilesystem(f"directory was substituted: {path}")

    # Opening and creating children.
    def open(
        self,
        name: str,
        *,
        directory: bool = True,
        expected: os.stat_result | None = None,
        same_device: bool = True,
    ) -> Pin:
        kind = "directory" if directory else "regular file"
        try:
            fd = os.open(leaf(name), _DIRECTORY if directory else _READ, dir_fd=self.fd)
        except FileNotFoundError as error:
            raise UnsafeFilesystem(f"filesystem entry disappeared: {name}") from error
        except OSError as error:
            if error.errno in _NOT_PLAIN:
                raise UnsafeFilesystem(f"entry is not a safe {kind}: {name}") from error
            owner = self.lstat(name) if error.errno in _DENIED else None
            if owner is not None and owner.st_uid == os.geteuid():
                raise ConfigurationError(f"{kind} is not readable by its owner: {name}") from error
            raise
        device = self.entry.st_dev if same_device else None
        return adopt(fd, name, device=device, expected=expected, regular=not directory)

    def mkdir(self, name: str, *, exist_ok: bool = False) -> Pin:
        """Create (or with `exist_ok`, reuse) an owner-only directory and pin it."""
        try:
            os.mkdir(leaf(name), 0o700, dir_fd=self.fd)
        except FileExistsError:
            if not exist_ok:
                raise
            return self.open(name)
        created: os.stat_result | None = None
        try:
            created = self.present(name)
            return self.open(name, expected=created)
        except BaseException:
            with suppress(OSError):  # rmdir only removes an empty directory
                if created is None or same(
                    os.stat(name, dir_fd=self.fd, follow_symlinks=False), created
                ):
                    os.rmdir(name, dir_fd=self.fd)
            raise

    def unique(self, prefix: str) -> tuple[str, Pin]:
        for _ in range(128):
            name = f"{prefix}{os.urandom(12).hex()}"
            with suppress(FileExistsError):
                return name, self.mkdir(name)
        raise UnsafeFilesystem(f"could not allocate a unique private directory: {prefix}*")


def adopt(
    fd: int,
    name: object,
    *,
    device: int | None = None,
    expected: os.stat_result | None = None,
    regular: bool = False,
) -> Pin:
    """Take ownership of a fresh descriptor: verify it or close it. No path leaks it."""
    try:
        entry = os.fstat(fd)
        if device is not None and entry.st_dev != device:
            raise CrossDeviceError(f"entry crosses a filesystem boundary: {name}")
        if expected is not None and not same(entry, expected):
            raise UnsafeFilesystem(f"entry was substituted before it was opened: {name}")
        if regular and not stat.S_ISREG(entry.st_mode):
            raise UnsafeFilesystem(f"entry is not a regular file: {name}")
    except BaseException:
        os.close(fd)
        raise
    return Pin(fd, entry)


def open_path(path: StrPath) -> Pin:
    """Pin an existing directory by path; its final component must not be a symlink."""
    text = fspath(path)
    try:
        fd = os.open(text, _DIRECTORY)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise UnsafeFilesystem(f"not a safe directory: {text}") from error
        raise
    return adopt(fd, text)


def read(file: Pin) -> bytes:
    """Read a pinned regular file completely, never more than one byte beyond its pinned size.

    The caller bounds `file.entry.st_size` first. A file that grew or shrank is `UnsafeFilesystem`.
    """
    chunks: list[bytes] = []
    remaining = file.entry.st_size + 1
    while remaining and (chunk := os.read(file.fd, min(remaining, _CHUNK))):
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b"".join(chunks)
    if len(data) != file.entry.st_size:
        raise UnsafeFilesystem("file changed while it was being read")
    return data


# -- locks -----------------------------------------------------------------------------------
def lock_file(directory: Pin, name: str, operation: int, what: str) -> Pin:
    """Open (creating, 0600) the owner-only regular file `name`, `flock` it, and pin it.

    After locking, `name` must still bind the locked inode, so a replaced lock file can never split
    its holders. Closing the pin releases the lock. `LOCK_NB` conflicts raise `BlockingIOError`.
    """
    try:
        fd = os.open(leaf(name), _LOCK, 0o600, dir_fd=directory.fd)
    except OSError as error:
        if error.errno in _NOT_PLAIN:
            raise UnsafeFilesystem(f"{what} is not a plain file") from error
        raise
    lock = adopt(fd, name, device=directory.entry.st_dev, regular=True)
    try:
        require_owner_only(lock.entry, what)
        fcntl.flock(lock.fd, operation)
        current = directory.lstat(name)
        if current is None or not same(current, lock.entry):
            raise UnsafeFilesystem(f"{what} was replaced while it was being locked")
    except BaseException:
        lock.close()
        raise
    return lock


@contextmanager
def locked(directory: Pin) -> Generator[None]:
    """Hold an exclusive `flock` coordinating entries of `directory`; released on exit.

    A dedicated read-only handle on the directory is locked, so closing it releases the lock.
    Where the filesystem refuses `flock` on such a handle (NFS: EBADF, ENOLCK, EOPNOTSUPP,
    EINVAL), the owner-only hidden file `.servatus.lock` in the directory is created and locked
    instead. It is never removed: unlinking a lock file would split its holders. Actors on one
    filesystem therefore always take the same path. `UnsupportedPlatform` if neither works.
    """
    fd = os.open(".", _DIRECTORY, dir_fd=directory.fd)
    with adopt(fd, ".", expected=directory.entry) as handle:
        try:
            fcntl.flock(handle.fd, fcntl.LOCK_EX)
        except OSError as error:
            if error.errno not in _NO_DIRECTORY_LOCK:
                raise
        else:
            yield
            return
    try:
        lock = lock_file(directory, LOCK_NAME, fcntl.LOCK_EX, f"directory lock {LOCK_NAME}")
    except OSError as error:
        if error.errno not in _NO_DIRECTORY_LOCK:
            raise
        raise UnsupportedPlatform("directory lock is unavailable on this filesystem") from error
    with lock:
        yield


# -- trees -----------------------------------------------------------------------------------
Visit: TypeAlias = Callable[[Pin, str, os.stat_result], None]
Order: TypeAlias = Callable[[str], "tuple[int, str]"]


def walk(directory: Pin, visit: Visit, *, strict: bool = True, order: Order | None = None) -> None:
    """Visit each entry once (by name, or `order`) without following links.

    `strict` admits only regular files and directories.
    """
    for name in sorted(os.listdir(directory.fd), key=order):
        entry = directory.present(name)
        if entry.st_dev != directory.entry.st_dev:
            raise CrossDeviceError(f"entry crosses a filesystem boundary: {name}")
        if strict and not (stat.S_ISDIR(entry.st_mode) or stat.S_ISREG(entry.st_mode)):
            raise UnsafeFilesystem(f"tree contains a symlink or special file: {name}")
        visit(directory, name, entry)


def sync_tree(directory: Pin, mode: int) -> None:
    """Sync every file, then set each directory to `mode` and sync it (children first)."""

    def visit(parent: Pin, name: str, entry: os.stat_result) -> None:
        if stat.S_ISDIR(entry.st_mode):
            with parent.open(name, expected=entry) as child:
                sync_tree(child, mode)
        else:
            with parent.open(name, directory=False, expected=entry) as child:
                sync(child.fd)
        parent.expect(name, entry)

    walk(directory, visit)
    os.fchmod(directory.fd, mode)
    sync(directory.fd)


def remove(
    parent: Pin, name: str, target: os.stat_result | Pin, *, order: Order | None = None
) -> None:
    """Remove exactly the pinned file or tree. A moved or substituted name is left alone.

    Directories are made owner-accessible (u+rwx) before they are opened and emptied, so private
    trees with read-only or unreadable directories can be reclaimed. A `Pin` target is used and
    left open. `order` sorts the target's own entries (default: by name).
    """
    expected = target.entry if isinstance(target, Pin) else target
    if not stat.S_ISDIR(expected.st_mode):
        parent.expect(name, expected)
        os.unlink(name, dir_fd=parent.fd)
        return
    with ExitStack() as stack:
        if isinstance(target, Pin):
            directory = target
        else:
            directory = stack.enter_context(_open_removable(parent, name, expected))
        parent.expect(name, expected)  # a moved pinned tree is preserved, not emptied
        mode = stat.S_IMODE(os.fstat(directory.fd).st_mode)
        if mode & 0o700 != 0o700:
            os.fchmod(directory.fd, mode | 0o700)
        walk(directory, remove, strict=False, order=order)
        parent.expect(name, expected)
        os.rmdir(name, dir_fd=parent.fd)


def _open_removable(parent: Pin, name: str, expected: os.stat_result) -> Pin:
    mode = stat.S_IMODE(expected.st_mode)
    if mode & 0o500 != 0o500:  # opening needs owner read and search; the inode is checked after
        parent.expect(name, expected)
        os.chmod(name, mode | 0o700, dir_fd=parent.fd, follow_symlinks=not _CHMOD_NOFOLLOW)
    return parent.open(name, expected=expected)


def discard(parent: Pin, name: str, target: os.stat_result | Pin) -> bool:
    """Best-effort unsynced `remove`; reports success. An absent name counts as removed."""
    try:
        if parent.lstat(name) is not None:
            remove(parent, name, target)
    except Exception:
        return False
    return True


def probe_umask(directory: Pin) -> int:
    """Read the effective creation mask without the racy process-wide `os.umask` round trip."""
    name = f".servatus-umask-{os.urandom(8).hex()}"
    fd = os.open(name, _NEW, 0o777, dir_fd=directory.fd)
    try:
        try:
            created = os.fstat(fd)
        finally:
            os.close(fd)
    finally:
        os.unlink(name, dir_fd=directory.fd)
    return 0o777 & ~stat.S_IMODE(created.st_mode)


# -- writing ---------------------------------------------------------------------------------
def write_new(dir_fd: int, name: str, data: bytes, *, mode: int = 0o600) -> os.stat_result:
    """Create `name` exclusively with exactly `mode`, write all bytes, and fsync the file.

    The directory is not synced; the caller decides how the file is installed. On any failure,
    including a failing close, the new file is removed (unsynced) and the error propagates.
    """
    checked = check_mode(mode)
    fd = os.open(leaf(name), _NEW, checked, dir_fd=dir_fd)
    created: os.stat_result | None = None
    try:
        try:
            created = os.fstat(fd)
            os.fchmod(fd, checked)
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError(errno.EIO, f"write made no progress: {name}")
                view = view[written:]
            sync(fd)
            return os.fstat(fd)
        finally:
            os.close(fd)  # inside the cleanup scope: a failing close still removes the file
    except BaseException:
        _unlink_created(dir_fd, name, created)
        raise


def replace_file(dir_fd: int, name: str, data: bytes, *, mode: int = 0o600) -> None:
    """Atomically replace (or create) `name` in `dir_fd` with `data`, durably.

    Stages a synced sibling, verifies it, renames it over `name`, then syncs the directory. Readers
    see the old or the new file, never a mixture. If the final directory sync fails the new file
    may already be visible; the error still propagates.
    """
    leaf(name)
    stage = f".servatus-replace-{os.urandom(12).hex()}"
    created = write_new(dir_fd, stage, data, mode=mode)
    try:
        current = os.stat(stage, dir_fd=dir_fd, follow_symlinks=False)
        if not same(current, created):
            raise UnsafeFilesystem(f"replacement stage was substituted: {stage}")
        os.rename(stage, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    except BaseException:
        _unlink_created(dir_fd, stage, created)
        raise
    sync(dir_fd)


def _unlink_created(dir_fd: int, name: str, created: os.stat_result | None) -> None:
    """Unlink `name` only if it is still the file this process created (exclusively)."""
    with suppress(OSError):
        if created is not None:
            current = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
            if not same(current, created):
                return
        os.unlink(name, dir_fd=dir_fd)


# -- no-replace commit -----------------------------------------------------------------------
def commit(source: Pin, name: str, target: Pin, destination: str, expected: os.stat_result) -> None:
    """Install `source/name` (which must still be `expected`) as `target/destination`.

    Never replaces and does not sync. Native: Linux `renameat2(RENAME_NOREPLACE)`, macOS
    `renameatx_np(RENAME_EXCL)`. On Linux only, when the call is unavailable (missing symbol,
    EINVAL, ENOSYS, EOPNOTSUPP), a regular file is hard-linked (its source name survives; the
    caller discards it) and a directory is renamed under an exclusive lock on an owner-controlled
    parent (see `locked`), released before the caller's parent sync.
    """
    current = source.expect(name, expected)
    native = native_noreplace()
    code = errno.ENOSYS
    if native is not None:
        code = native(source.fd, os.fsencode(name), target.fd, os.fsencode(destination))
        if code == 0:
            return
    if not (sys.platform.startswith("linux") and code in _UNSUPPORTED):
        raise rename_error(code)
    if stat.S_ISREG(current.st_mode):
        try:
            os.link(
                name, destination, src_dir_fd=source.fd, dst_dir_fd=target.fd, follow_symlinks=False
            )
        except OSError as error:
            raise rename_error(error.errno) from error
        if not same(target.present(destination), expected):
            raise UnsafeFilesystem(f"published file is not the verified source: {destination}")
    elif stat.S_ISDIR(current.st_mode):
        _locked_rename(source, name, target, destination, current)
    else:
        raise UnsafeFilesystem(f"publication source is not a regular file or directory: {name}")


def _locked_rename(
    source: Pin, name: str, target: Pin, destination: str, expected: os.stat_result
) -> None:
    with locked(target):
        owner = os.fstat(target.fd)
        if owner.st_uid != os.geteuid() or owner.st_mode & 0o022:
            raise UnsafeFilesystem("fallback requires an owner-controlled publication parent")
        source.expect(name, expected)
        target.absent(destination, destination)
        os.rename(name, destination, src_dir_fd=source.fd, dst_dir_fd=target.fd)
        if not same(target.present(destination), expected):
            raise UnsafeFilesystem(f"published directory is not the verified source: {destination}")


@cache
def native_noreplace() -> NoReplace | None:
    """Look up the atomic no-replace rename once; None means this libc or kernel lacks it."""
    library = ctypes.CDLL(None, use_errno=True)
    # A value rather than a static condition, so type checkers analyse both platforms.
    darwin = sys.platform == "darwin"
    function: Any = getattr(library, "renameatx_np" if darwin else "renameat2", None)
    if darwin or function is not None:
        return _bind(function, (), _RENAME_EXCL if darwin else _RENAME_NOREPLACE)
    number = _SYS_RENAMEAT2.get(platform.machine()) if os.uname().sysname == "Linux" else None
    syscall: Any = getattr(library, "syscall", None)
    if number is None or syscall is None:
        return None  # e.g. glibc < 2.28 on an unknown architecture: use the documented fallback
    return _bind(syscall, (number,), _RENAME_NOREPLACE)


def _bind(function: Any, prefix: tuple[int, ...], flag: int) -> NoReplace | None:
    if function is None:
        return None
    function.argtypes = (
        *(ctypes.c_long for _ in prefix),
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_long if prefix else ctypes.c_int

    def call(source_fd: int, source: bytes, target_fd: int, target: bytes) -> int:
        if function(*prefix, source_fd, source, target_fd, target, flag) == 0:
            return 0
        return ctypes.get_errno() or errno.EIO

    return call


def rename_error(code: int | None) -> Exception:
    if code in {errno.EEXIST, errno.ENOTEMPTY}:
        return DestinationExists("publication destination already exists")
    if code in _UNSUPPORTED:
        return UnsupportedPlatform("filesystem does not support an atomic no-replace rename")
    code = code or errno.EIO
    return os_error(OSError(code, os.strerror(code)), "publication commit")
