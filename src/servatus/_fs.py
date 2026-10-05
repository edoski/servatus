"""Pinned-descriptor POSIX primitives shared by publication and the campaign store.

Nothing here knows about publication policy. Every descriptor is wrapped in a `Pin` the moment it
is opened: the pin is verified (device, inode, type) or the descriptor is closed before the error
propagates, so no failure path leaks a descriptor. Names are always resolved relative to a pinned
directory and never followed.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import platform
import stat
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext, suppress
from functools import cache
from pathlib import Path
from typing import Any, TypeAlias

from .errors import (
    ConfigurationError,
    CrossDeviceError,
    DestinationExists,
    UnsafeFilesystem,
    UnsupportedPlatform,
)

StrPath: TypeAlias = "str | os.PathLike[str]"
NoReplace: TypeAlias = Callable[[int, bytes, int, bytes], int]
"""A native no-replace rename returning 0 or an errno value."""

_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
_READ = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_NEW = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
_UNSUPPORTED = frozenset({errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP})
_RENAME_NOREPLACE = 0x1  # Linux renameat2 flag
_RENAME_EXCL = 0x4  # macOS renameatx_np flag
_SYS_RENAMEAT2 = {"x86_64": 316, "amd64": 316, "aarch64": 276, "arm64": 276}


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
        raise UnsafeFilesystem(f"{what} is not owned by this user with owner-only permissions")


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


def try_sync(fd: int) -> bool:
    try:
        sync(fd)
    except Exception:
        return False
    return True


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
        except OSError as error:
            raise UnsafeFilesystem(f"filesystem entry is unavailable: {name}") from error

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
        self, name: str, *, directory: bool = True, expected: os.stat_result | None = None
    ) -> Pin:
        try:
            fd = os.open(leaf(name), _DIRECTORY if directory else _READ, dir_fd=self.fd)
        except OSError as error:
            kind = "directory" if directory else "regular file"
            raise UnsafeFilesystem(f"entry is not a safe {kind}: {name}") from error
        return adopt(fd, name, device=self.entry.st_dev, expected=expected, regular=not directory)

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
        raise UnsafeFilesystem(f"not a safe directory: {text}") from error
    return adopt(fd, text)


# -- trees -----------------------------------------------------------------------------------
Visit: TypeAlias = Callable[[Pin, str, os.stat_result], None]


def walk(directory: Pin, visit: Visit, *, strict: bool = True) -> None:
    """Visit each entry once without following links; `strict` admits only files and dirs."""
    for name in sorted(os.listdir(directory.fd)):
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


def remove(parent: Pin, name: str, expected: os.stat_result, *, pinned: Pin | None = None) -> None:
    """Remove exactly the pinned file or tree. A moved or substituted name is left alone.

    Directories being removed are made owner-writable first, so read-only private trees can be
    reclaimed; `pinned` (when given) is used and left open.
    """
    if not stat.S_ISDIR(expected.st_mode):
        parent.expect(name, expected)
        os.unlink(name, dir_fd=parent.fd)
        return
    opened: AbstractContextManager[Pin] = (
        nullcontext(pinned) if pinned is not None else parent.open(name, expected=expected)
    )
    with opened as directory:
        current = os.fstat(directory.fd)
        if not same(current, expected):
            raise UnsafeFilesystem(f"cleanup target was substituted: {name}")
        parent.expect(name, expected)  # a moved pinned tree is preserved, not emptied
        if stat.S_IMODE(current.st_mode) & 0o700 != 0o700:
            os.fchmod(directory.fd, stat.S_IMODE(current.st_mode) | 0o700)
        walk(directory, remove, strict=False)
        parent.expect(name, expected)
        os.rmdir(name, dir_fd=parent.fd)


def discard(parent: Pin, name: str, expected: os.stat_result, *, pinned: Pin | None = None) -> bool:
    """Best-effort unsynced `remove`; reports success. An absent name counts as removed."""
    try:
        if parent.lstat(name) is not None:
            remove(parent, name, expected, pinned=pinned)
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

    The directory is not synced; the caller decides how the file is installed. On failure the new
    file is removed (unsynced) and the error propagates.
    """
    checked = check_mode(mode)
    fd = os.open(leaf(name), _NEW, checked, dir_fd=dir_fd)
    created: os.stat_result | None = None
    try:
        created = os.fstat(fd)
        os.fchmod(fd, checked)
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view) :]
        sync(fd)
        return os.fstat(fd)
    except BaseException:
        _unlink_created(dir_fd, name, created)
        raise
    finally:
        os.close(fd)


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
    with suppress(OSError):
        if created is not None:
            current = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
            if not same(current, created):
                return
        os.unlink(name, dir_fd=dir_fd)


# -- no-replace commit -----------------------------------------------------------------------
def commit(source: Pin, name: str, target: Pin, destination: str, expected: os.stat_result) -> None:
    """Install `source/name` as `target/destination` without ever replacing. Does not sync.

    Native: Linux `renameat2(RENAME_NOREPLACE)`, macOS `renameatx_np(RENAME_EXCL)`. On Linux only,
    when the call is unavailable (missing symbol, EINVAL, ENOSYS, EOPNOTSUPP), a regular file is
    hard-linked (its source name survives; the caller discards it) and a directory is renamed under
    an exclusive `flock` on an owner-controlled parent, released before the caller's parent sync.
    """
    native = native_noreplace()
    code = errno.ENOSYS
    if native is not None:
        code = native(source.fd, os.fsencode(name), target.fd, os.fsencode(destination))
        if code == 0:
            return
    if not (sys.platform.startswith("linux") and code in _UNSUPPORTED):
        raise rename_error(code)
    current = source.expect(name, expected)
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
    try:
        fd = os.open(".", _DIRECTORY, dir_fd=target.fd)
    except OSError as error:
        raise UnsupportedPlatform("publication parent lock is unavailable") from error
    with adopt(fd, ".") as lock:  # a dedicated handle: closing it releases the lock
        if not same(lock.entry, target.entry):
            raise UnsafeFilesystem("publication parent lock refers to another directory")
        try:
            fcntl.flock(lock.fd, fcntl.LOCK_EX)
        except OSError as error:
            raise UnsupportedPlatform("publication parent lock is unavailable") from error
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
    if _darwin():
        return _bind(getattr(library, "renameatx_np", None), (), _RENAME_EXCL)
    function: Any = getattr(library, "renameat2", None)
    if function is not None:
        return _bind(function, (), _RENAME_NOREPLACE)
    number = _SYS_RENAMEAT2.get(platform.machine()) if os.uname().sysname == "Linux" else None
    syscall: Any = getattr(library, "syscall", None)
    if number is None or syscall is None:
        return None  # e.g. glibc < 2.28 on an unknown architecture: use the documented fallback
    return _bind(syscall, (number,), _RENAME_NOREPLACE)


def _darwin() -> bool:
    return sys.platform == "darwin"


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
    if code == errno.EXDEV:
        return CrossDeviceError("publication commit crosses a filesystem boundary")
    if code in _UNSUPPORTED:
        return UnsupportedPlatform("filesystem does not support an atomic no-replace rename")
    code = code or errno.EIO
    return OSError(code, os.strerror(code))
