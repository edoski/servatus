from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import stat
import sys
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from ._errors import (
    CrossDevicePublication,
    DestinationExists,
    PublicationError,
    UnsafePublication,
    UnsupportedPlatform,
)

_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
_FILE_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_FILE_STAGE_FLAGS = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW


class _NoreplaceUnavailable(Exception):
    pass


@dataclass(frozen=True, slots=True)
class _CommitOutcome:
    cleanup_pending: bool


@dataclass(frozen=True, slots=True)
class _TransactionOutcome:
    destination: Path
    cleanup_pending: bool


def require_supported_platform() -> None:
    if not (sys.platform.startswith("linux") or sys.platform == "darwin"):
        raise UnsupportedPlatform(f"unsupported platform: {sys.platform}")


def normalize_destination(destination: Path) -> tuple[Path, str]:
    raw = Path(destination)
    reject_nul_path(raw)
    if not raw.name or raw.name in {".", ".."}:
        raise UnsafePublication("destination must name an entry inside an existing parent")
    try:
        parent = raw.parent.resolve(strict=True)
    except (OSError, ValueError) as error:
        raise UnsafePublication("destination parent does not exist") from error
    if not parent.is_dir():
        raise UnsafePublication("destination parent is not a directory")
    return parent, raw.name


def open_directory(path: Path) -> int:
    reject_nul_path(path)
    try:
        return os.open(path, _DIRECTORY_FLAGS)
    except OSError as error:
        raise UnsafePublication(f"unsafe directory: {path}") from error


def open_directory_at(parent_fd: int, name: str) -> int:
    validate_leaf(name)
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as error:
        raise UnsafePublication(f"unsafe directory entry: {name}") from error
    parent_device = os.fstat(parent_fd).st_dev
    if os.fstat(descriptor).st_dev != parent_device:
        os.close(descriptor)
        raise CrossDevicePublication(f"directory crosses a filesystem boundary: {name}")
    return descriptor


def same_entry(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def ensure_entry(parent_fd: int, name: str, expected: os.stat_result) -> None:
    validate_leaf(name)
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise UnsafePublication(f"filesystem entry disappeared: {name}") from error
    if not same_entry(current, expected):
        raise UnsafePublication(f"filesystem entry changed during publication: {name}")


def ensure_directory_path(path: Path, descriptor: int) -> None:
    reject_nul_path(path)
    try:
        current = path.stat(follow_symlinks=False)
    except OSError as error:
        raise UnsafePublication(f"publication parent disappeared: {path}") from error
    if not stat.S_ISDIR(current.st_mode) or not same_entry(current, os.fstat(descriptor)):
        raise UnsafePublication(f"publication parent changed: {path}")


def make_directory_at(parent_fd: int, name: str) -> tuple[int, os.stat_result]:
    validate_leaf(name)
    created: os.stat_result | None = None
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    else:
        try:
            created = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as error:
            raise UnsafePublication(f"new directory is unavailable: {name}") from error
    descriptor = open_directory_at(parent_fd, name)
    try:
        opened = os.fstat(descriptor)
        if created is not None and not same_entry(created, opened):
            raise UnsafePublication(f"new directory changed before open: {name}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, opened


def make_unique_stage(parent_fd: int, prefix: str) -> tuple[str, int, os.stat_result]:
    validate_leaf(prefix)
    for _ in range(128):
        name = f"{prefix}{os.urandom(12).hex()}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = -1
        try:
            descriptor = open_directory_at(parent_fd, name)
            opened = os.fstat(descriptor)
            if not same_entry(entry, opened):
                raise UnsafePublication(f"new publication stage changed before open: {name}")
        except BaseException as error:
            cleanup_after_failure(parent_fd, name, entry, error)
            if descriptor >= 0:
                os.close(descriptor)
            raise
        return name, descriptor, opened
    raise PublicationError("could not allocate a unique publication stage")


def make_unique_file_stage(parent_fd: int, prefix: str) -> tuple[str, int, os.stat_result]:
    validate_leaf(prefix)
    parent_device = os.fstat(parent_fd).st_dev
    for _ in range(128):
        name = f"{prefix}{os.urandom(12).hex()}"
        try:
            descriptor = os.open(
                name,
                _FILE_STAGE_FLAGS,
                0o666,
                dir_fd=parent_fd,
            )
        except FileExistsError:
            continue
        entry = os.fstat(descriptor)
        try:
            if not stat.S_ISREG(entry.st_mode):
                raise UnsafePublication("publication stage is not a regular file")
            if entry.st_dev != parent_device:
                raise CrossDevicePublication("publication stage crosses a filesystem boundary")
        except BaseException as error:
            _cleanup_entry_after_failure(parent_fd, name, entry, error, remove_file_at)
            os.close(descriptor)
            raise
        return name, descriptor, entry
    raise PublicationError("could not allocate a unique publication stage")


def sync_descriptor(descriptor: int) -> None:
    os.fsync(descriptor)


def sync_tree(descriptor: int, device: int) -> None:
    for name in os.listdir(descriptor):
        try:
            entry = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError as error:
            raise UnsafePublication(f"draft entry disappeared: {name}") from error
        if entry.st_dev != device:
            raise CrossDevicePublication(f"draft entry crosses a filesystem boundary: {name}")
        if stat.S_ISREG(entry.st_mode):
            _sync_regular_file(descriptor, name, entry)
        elif stat.S_ISDIR(entry.st_mode):
            child = open_directory_at(descriptor, name)
            try:
                sync_tree(child, device)
            finally:
                os.close(child)
            ensure_entry(descriptor, name, entry)
        else:
            raise UnsafePublication(f"draft contains a symlink or special file: {name}")
    sync_descriptor(descriptor)


def _sync_regular_file(parent_fd: int, name: str, expected: os.stat_result) -> None:
    try:
        descriptor = os.open(name, _FILE_FLAGS, dir_fd=parent_fd)
    except OSError as error:
        raise UnsafePublication(f"unsafe regular file: {name}") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not same_entry(opened, expected):
            raise UnsafePublication(f"draft file changed during publication: {name}")
        sync_descriptor(descriptor)
        synced = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not same_entry(synced, current):
            raise UnsafePublication(f"draft file changed during publication: {name}")
        if (opened.st_size, opened.st_mtime_ns) != (synced.st_size, synced.st_mtime_ns):
            raise UnsafePublication(f"draft file was modified during publication: {name}")
    finally:
        os.close(descriptor)


def sync_file_stage(
    parent_fd: int,
    name: str,
    descriptor: int,
    expected: os.stat_result,
) -> None:
    opened = os.fstat(descriptor)
    parent_device = os.fstat(parent_fd).st_dev
    if opened.st_dev != parent_device or expected.st_dev != parent_device:
        raise CrossDevicePublication("publication stage crosses a filesystem boundary")
    if not stat.S_ISREG(opened.st_mode) or not same_entry(opened, expected):
        raise UnsafePublication("publication stage changed during publication")
    ensure_entry(parent_fd, name, opened)
    sync_descriptor(descriptor)
    synced = os.fstat(descriptor)
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise UnsafePublication("publication stage disappeared") from error
    if (
        not stat.S_ISREG(synced.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or not same_entry(synced, expected)
        or not same_entry(synced, current)
    ):
        raise UnsafePublication("publication stage changed during publication")
    if (opened.st_size, opened.st_mtime_ns) != (synced.st_size, synced.st_mtime_ns):
        raise UnsafePublication("publication stage was modified during publication")


def reject_nul_path(path: Path) -> None:
    if "\0" in os.fspath(path):
        raise UnsafePublication(f"path contains an embedded NUL: {path!r}")


def validate_leaf(name: str) -> None:
    if not name or name in {".", ".."} or "/" in name or "\0" in name:
        raise UnsafePublication(f"unsafe filesystem leaf: {name!r}")


def commit_noreplace(
    parent_fd: int,
    source: str,
    destination: str,
    expected_source: os.stat_result,
) -> _CommitOutcome:
    require_supported_platform()
    validate_leaf(source)
    validate_leaf(destination)
    expected_parent = os.fstat(parent_fd)
    if sys.platform.startswith("linux"):
        try:
            _linux_rename_noreplace(parent_fd, source, destination)
        except _NoreplaceUnavailable:
            _verify_fallback_parent(parent_fd, expected_parent)
            current_source = _verified_source(parent_fd, source, expected_source)
            if stat.S_ISREG(current_source.st_mode):
                return _link_file_noreplace(parent_fd, source, destination, current_source)
            if stat.S_ISDIR(current_source.st_mode):
                _locked_directory_noreplace(
                    parent_fd,
                    source,
                    destination,
                    expected_parent,
                    current_source,
                )
                return _CommitOutcome(cleanup_pending=False)
            raise UnsafePublication(
                "publication source is not a regular file or directory"
            ) from None
        sync_descriptor(parent_fd)
        return _CommitOutcome(cleanup_pending=False)
    _macos_rename_noreplace(parent_fd, source, destination)
    sync_descriptor(parent_fd)
    return _CommitOutcome(cleanup_pending=False)


def _verified_source(
    parent_fd: int,
    source: str,
    expected: os.stat_result,
) -> os.stat_result:
    try:
        current = os.stat(source, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise UnsafePublication(f"publication source is unavailable: {source}") from error
    if not same_entry(current, expected):
        raise UnsafePublication(f"publication source changed during commit: {source}")
    return current


def _verify_fallback_parent(parent_fd: int, expected: os.stat_result) -> os.stat_result:
    current = os.fstat(parent_fd)
    if not stat.S_ISDIR(current.st_mode) or not same_entry(current, expected):
        raise UnsafePublication("publication parent changed before fallback")
    return current


def _verify_owner_controlled_parent(parent_fd: int, expected: os.stat_result) -> None:
    current = _verify_fallback_parent(parent_fd, expected)
    if current.st_uid != os.geteuid() or current.st_mode & 0o022:
        raise UnsafePublication("fallback requires an owner-controlled publication parent")


def _link_file_noreplace(
    parent_fd: int,
    source: str,
    destination: str,
    expected_source: os.stat_result,
) -> _CommitOutcome:
    try:
        os.link(
            source,
            destination,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileExistsError as error:
        raise DestinationExists("publication destination already exists") from error
    except OSError as error:
        if error.errno == errno.EXDEV:
            raise CrossDevicePublication(
                "publication commit crosses a filesystem boundary"
            ) from error
        raise
    linked = os.stat(destination, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISREG(linked.st_mode) or not same_entry(linked, expected_source):
        raise UnsafePublication("published file does not match the verified source")
    sync_descriptor(parent_fd)
    try:
        remove_file_at(parent_fd, source, expected_source)
    except Exception:
        return _reconcile_file_cleanup(parent_fd, source)
    return _CommitOutcome(cleanup_pending=False)


def _reconcile_file_cleanup(parent_fd: int, source: str) -> _CommitOutcome:
    try:
        os.stat(source, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        try:
            sync_descriptor(parent_fd)
        except Exception:
            return _CommitOutcome(cleanup_pending=True)
        return _CommitOutcome(cleanup_pending=False)
    except OSError:
        pass
    return _CommitOutcome(cleanup_pending=True)


def _locked_directory_noreplace(
    parent_fd: int,
    source: str,
    destination: str,
    expected_parent: os.stat_result,
    expected_source: os.stat_result,
) -> None:
    try:
        lock_fd = os.open(".", _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as error:
        raise UnsupportedPlatform("publication parent lock is unavailable") from error
    try:
        _verify_fallback_parent(lock_fd, expected_parent)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        except OSError as error:
            raise UnsupportedPlatform("publication parent lock is unavailable") from error
        _verify_owner_controlled_parent(parent_fd, expected_parent)
        current_source = _verified_source(parent_fd, source, expected_source)
        if not stat.S_ISDIR(current_source.st_mode):
            raise UnsafePublication("publication source is not a directory")
        try:
            os.stat(destination, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError as error:
            raise UnsafePublication("publication destination could not be inspected") from error
        else:
            raise DestinationExists("publication destination already exists")
        os.rename(source, destination, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        published = os.stat(destination, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(published.st_mode) or not same_entry(published, expected_source):
            raise UnsafePublication("published directory does not match the verified source")
        sync_descriptor(parent_fd)
    finally:
        with suppress(OSError):
            os.close(lock_fd)


def _linux_rename_noreplace(parent_fd: int, source: str, destination: str) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = library.renameat2
    except AttributeError as error:
        raise UnsupportedPlatform("libc does not expose renameat2") from error
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_fd,
        os.fsencode(source),
        parent_fd,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
            raise _NoreplaceUnavailable
        _raise_rename_error(error_number)


def _macos_rename_noreplace(parent_fd: int, source: str, destination: str) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    try:
        renameatx_np = library.renameatx_np
    except AttributeError as error:
        raise UnsupportedPlatform("libc does not expose renameatx_np") from error
    renameatx_np.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameatx_np.restype = ctypes.c_int
    result = renameatx_np(
        parent_fd,
        os.fsencode(source),
        parent_fd,
        os.fsencode(destination),
        _RENAME_EXCL,
    )
    if result != 0:
        _raise_rename_error(ctypes.get_errno())


def _raise_rename_error(error_number: int) -> None:
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise DestinationExists("publication destination already exists")
    if error_number == errno.EXDEV:
        raise CrossDevicePublication("publication commit crosses a filesystem boundary")
    if error_number in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
        raise UnsupportedPlatform("filesystem does not support atomic no-replace rename")
    raise OSError(error_number, os.strerror(error_number))


def remove_tree_at(parent_fd: int, name: str, expected: os.stat_result) -> None:
    _remove_directory_entry(parent_fd, name, expected, expected.st_dev)
    sync_descriptor(parent_fd)


def _remove_directory_entry(
    parent_fd: int,
    name: str,
    expected: os.stat_result,
    device: int,
) -> None:
    if not stat.S_ISDIR(expected.st_mode):
        raise UnsafePublication(f"cleanup target is not a directory: {name}")
    descriptor = open_directory_at(parent_fd, name)
    try:
        opened = os.fstat(descriptor)
        if not same_entry(opened, expected):
            raise UnsafePublication(f"cleanup target changed before open: {name}")
        ensure_entry(parent_fd, name, expected)
        _remove_tree_contents(descriptor, device)
        ensure_entry(parent_fd, name, expected)
        os.rmdir(name, dir_fd=parent_fd)
    finally:
        os.close(descriptor)


def _remove_tree_contents(descriptor: int, device: int) -> None:
    for name in sorted(os.listdir(descriptor)):
        try:
            entry = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError as error:
            raise UnsafePublication(f"cleanup entry disappeared: {name}") from error
        if entry.st_dev != device:
            raise CrossDevicePublication(f"cleanup entry crosses a filesystem boundary: {name}")
        if stat.S_ISDIR(entry.st_mode):
            _remove_directory_entry(descriptor, name, entry, device)
        else:
            _remove_nondirectory_entry(descriptor, name, entry)


def _remove_nondirectory_entry(
    parent_fd: int,
    name: str,
    expected: os.stat_result,
) -> None:
    if stat.S_ISREG(expected.st_mode):
        _pin_and_remove_regular_file(parent_fd, name, expected)
        return
    ensure_entry(parent_fd, name, expected)
    os.unlink(name, dir_fd=parent_fd)


def _pin_and_remove_regular_file(
    parent_fd: int,
    name: str,
    expected: os.stat_result,
) -> None:
    pin = ""
    for _ in range(128):
        candidate = f".servatus-cleanup-{os.urandom(12).hex()}"
        try:
            os.link(
                name,
                candidate,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            continue
        except OSError as error:
            raise UnsafePublication(f"cleanup file could not be pinned: {name}") from error
        pin = candidate
        break
    if not pin:
        raise UnsafePublication("could not allocate a cleanup file pin")
    try:
        pinned = os.stat(pin, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise UnsafePublication(f"cleanup file pin is unavailable: {name}") from error
    try:
        if not stat.S_ISREG(pinned.st_mode) or not same_entry(pinned, expected):
            raise UnsafePublication(f"cleanup file changed before pin: {name}")
        ensure_entry(parent_fd, name, expected)
        os.unlink(name, dir_fd=parent_fd)
        ensure_entry(parent_fd, pin, pinned)
        os.unlink(pin, dir_fd=parent_fd)
    except BaseException as error:
        try:
            ensure_entry(parent_fd, pin, pinned)
            os.unlink(pin, dir_fd=parent_fd)
        except Exception as cleanup_error:
            error.add_note(f"Servatus could not remove cleanup file pin: {cleanup_error}")
        raise


def remove_file_at(parent_fd: int, name: str, expected: os.stat_result) -> None:
    ensure_entry(parent_fd, name, expected)
    if not stat.S_ISREG(expected.st_mode):
        raise UnsafePublication(f"cleanup target is not a regular file: {name}")
    os.unlink(name, dir_fd=parent_fd)
    sync_descriptor(parent_fd)


def cleanup_after_failure(
    parent_fd: int,
    name: str,
    expected: os.stat_result,
    error: BaseException,
) -> None:
    _cleanup_entry_after_failure(parent_fd, name, expected, error, remove_tree_at)


def _cleanup_entry_after_failure(
    parent_fd: int,
    name: str,
    expected: os.stat_result,
    error: BaseException,
    remove: Callable[[int, str, os.stat_result], None],
) -> None:
    try:
        remove(parent_fd, name, expected)
    except Exception as cleanup_error:
        error.add_note(f"Servatus could not remove failed stage: {cleanup_error}")


def publication_attempt(
    destination: Path,
    build: Callable[[Path, int, int], None],
) -> _TransactionOutcome:
    require_supported_platform()
    parent, destination_name = normalize_destination(destination)
    parent_fd = open_directory(parent)
    try:
        return publication_attempt_at(parent, parent_fd, destination_name, build)
    finally:
        os.close(parent_fd)


def publication_attempt_at(
    parent: Path,
    parent_fd: int,
    destination_name: str,
    build: Callable[[Path, int, int], None],
) -> _TransactionOutcome:
    return _publication_transaction_at(
        parent,
        parent_fd,
        destination_name,
        build,
        lambda: make_unique_stage(parent_fd, ".servatus-stage-"),
        lambda name, descriptor, entry: sync_tree(descriptor, entry.st_dev),
        remove_tree_at,
    )


def file_publication_attempt(
    destination: Path,
    write: Callable[[Path], None],
) -> _TransactionOutcome:
    require_supported_platform()
    parent, destination_name = normalize_destination(destination)
    parent_fd = open_directory(parent)
    try:
        return file_publication_attempt_at(parent, parent_fd, destination_name, write)
    finally:
        os.close(parent_fd)


def file_publication_attempt_at(
    parent: Path,
    parent_fd: int,
    destination_name: str,
    write: Callable[[Path], None],
) -> _TransactionOutcome:
    def build(path: Path, descriptor: int, device: int) -> None:
        del descriptor, device
        write(path)

    return _publication_transaction_at(
        parent,
        parent_fd,
        destination_name,
        build,
        lambda: make_unique_file_stage(parent_fd, ".servatus-file-stage-"),
        lambda name, descriptor, entry: sync_file_stage(parent_fd, name, descriptor, entry),
        remove_file_at,
    )


def _publication_transaction_at(
    parent: Path,
    parent_fd: int,
    destination_name: str,
    build: Callable[[Path, int, int], None],
    make_stage: Callable[[], tuple[str, int, os.stat_result]],
    sync_stage: Callable[[str, int, os.stat_result], None],
    remove_stage: Callable[[int, str, os.stat_result], None],
) -> _TransactionOutcome:
    ensure_directory_path(parent, parent_fd)
    stage_name = ""
    stage_fd = -1
    stage_entry: os.stat_result | None = None
    try:
        stage_name, stage_fd, stage_entry = make_stage()
        build(parent / stage_name, stage_fd, stage_entry.st_dev)
        sync_stage(stage_name, stage_fd, stage_entry)
        ensure_entry(parent_fd, stage_name, stage_entry)
        ensure_directory_path(parent, parent_fd)
        commit = commit_noreplace(
            parent_fd,
            stage_name,
            destination_name,
            stage_entry,
        )
        return _TransactionOutcome(parent / destination_name, commit.cleanup_pending)
    except BaseException as error:
        if stage_name and stage_entry is not None:
            _cleanup_entry_after_failure(parent_fd, stage_name, stage_entry, error, remove_stage)
        raise
    finally:
        if stage_fd >= 0:
            with suppress(OSError):
                os.close(stage_fd)
