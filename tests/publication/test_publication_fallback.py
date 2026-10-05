from __future__ import annotations

import ctypes
import errno
import fcntl
import multiprocessing
import os
import platform
import stat
import sys
import threading
from collections.abc import Callable
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, NamedTuple, NoReturn

import pytest

from servatus import _fs
from servatus.errors import (
    CrossDeviceError,
    DestinationExists,
    Unavailable,
    UnsafeFilesystem,
    UnsupportedPlatform,
)
from servatus.publication import Draft, Workspace, publish, publish_file


class FakeFunction:
    """A libc symbol stand-in that records calls and reports errors through ctypes errno."""

    def __init__(self, behaviour: Callable[..., int]) -> None:
        self.behaviour = behaviour
        self.calls: list[tuple[Any, ...]] = []
        self.argtypes: object = None
        self.restype: object = None

    def __call__(self, *args: Any) -> int:
        self.calls.append(args)
        return self.behaviour(*args)


class FakeLibc:
    def __init__(self, **symbols: FakeFunction) -> None:
        for name, function in symbols.items():
            setattr(self, name, function)


def failing(code: int) -> Callable[..., int]:
    def fail(*args: Any) -> int:
        ctypes.set_errno(code)
        return -1

    return fail


def renaming(*args: Any) -> int:
    source_fd, source, target_fd, target = args[-5:-1]
    try:
        os.stat(target, dir_fd=target_fd, follow_symlinks=False)
    except FileNotFoundError:
        os.rename(source, target, src_dir_fd=source_fd, dst_dir_fd=target_fd)
        return 0
    ctypes.set_errno(errno.EEXIST)
    return -1


def use_libc(monkeypatch: pytest.MonkeyPatch, system: str, libc: object) -> None:
    monkeypatch.setattr(sys, "platform", system)

    def load(*args: object, **kwargs: object) -> object:
        return libc

    monkeypatch.setattr(ctypes, "CDLL", load)
    _fs.native_noreplace.cache_clear()


def no_symbols(*args: object, **kwargs: object) -> object:
    return object()


def forbidden(*args: object, **kwargs: object) -> NoReturn:
    pytest.fail("the Linux fallback must not run")


def force_fallback_in_this_process() -> None:
    sys.platform = "linux"
    setattr(ctypes, "CDLL", no_symbols)  # noqa: B010


def pinned_commit(parent: Path, source: str, destination: str) -> None:
    with _fs.open_path(parent) as directory:
        _fs.commit(directory, source, directory, destination, directory.present(source))


# -- native lookup ---------------------------------------------------------------------------
def test_missing_renameat2_symbol_takes_the_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: a libc without renameat2 (glibc < 2.28) used to be fatal.
    use_libc(monkeypatch, "linux", FakeLibc())

    assert _fs.native_noreplace() is None
    publication = publish_file(tmp_path / "a.json", lambda path: path.write_text("{}"))
    publish(tmp_path / "tree", lambda draft: (draft.path / "value").write_text("x"))

    assert publication.cleanup_pending is False
    assert (tmp_path / "a.json").read_text() == "{}"
    assert (tmp_path / "tree/value").read_text() == "x"


@pytest.mark.parametrize(("machine", "number"), [("x86_64", 316), ("aarch64", 276)])
def test_missing_symbol_uses_renameat2_syscall_on_linux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: str, number: int
) -> None:
    syscall = FakeFunction(renaming)

    class Uname(NamedTuple):
        sysname: str

    use_libc(monkeypatch, "linux", FakeLibc(syscall=syscall))
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(os, "uname", lambda: Uname("Linux"))

    publish_file(tmp_path / "result", lambda path: path.write_text("native"))

    ((called_number, *_, flag),) = syscall.calls
    assert (called_number, flag) == (number, 1)
    assert (tmp_path / "result").read_text() == "native"


def test_unknown_architecture_without_symbol_takes_the_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    syscall = FakeFunction(failing(errno.EPERM))

    class Uname(NamedTuple):
        sysname: str

    use_libc(monkeypatch, "linux", FakeLibc(syscall=syscall))
    monkeypatch.setattr(platform, "machine", lambda: "riscv64")
    monkeypatch.setattr(os, "uname", lambda: Uname("Linux"))

    publish(tmp_path / "result", lambda draft: None)

    assert syscall.calls == []
    assert (tmp_path / "result").is_dir()


def test_native_lookup_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    loads: list[object] = []

    def load(*args: object, **kwargs: object) -> FakeLibc:
        loads.append(args)
        return FakeLibc(renameat2=FakeFunction(renaming))

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", load)
    _fs.native_noreplace.cache_clear()

    assert _fs.native_noreplace() is _fs.native_noreplace()
    assert len(loads) == 1


def test_native_noreplace_remains_the_fast_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    renameat2 = FakeFunction(renaming)
    use_libc(monkeypatch, "linux", FakeLibc(renameat2=renameat2))
    monkeypatch.setattr(fcntl, "flock", forbidden)

    publish(tmp_path / "result", lambda draft: None)

    assert len(renameat2.calls) == 1
    assert renameat2.calls[0][-1] == 1  # RENAME_NOREPLACE
    assert (tmp_path / "result").is_dir()


@pytest.mark.parametrize(
    ("code", "expected", "message"),
    [
        (errno.EXDEV, CrossDeviceError, "filesystem boundary"),
        (errno.EIO, Unavailable, "Input/output error"),
        (errno.EEXIST, DestinationExists, "already exists"),
    ],
)
def test_other_native_errors_never_enter_the_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code: int,
    expected: type[Exception],
    message: str,
) -> None:
    use_libc(monkeypatch, "linux", FakeLibc(renameat2=FakeFunction(failing(code))))
    monkeypatch.setattr(os, "link", forbidden)
    monkeypatch.setattr(fcntl, "flock", forbidden)

    with pytest.raises(expected, match=message):
        publish_file(tmp_path / "file", lambda path: path.write_text("x"))
    with pytest.raises(expected, match=message):
        publish(tmp_path / "tree", lambda draft: None)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("code", [errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP])
def test_unsupported_native_errors_enter_the_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    use_libc(monkeypatch, "linux", FakeLibc(renameat2=FakeFunction(failing(code))))

    publish_file(tmp_path / "file", lambda path: path.write_text("x"))
    publish(tmp_path / "tree", lambda draft: None)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["file", "tree"]


def test_macos_without_renameatx_np_is_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_libc(monkeypatch, "darwin", FakeLibc())

    with pytest.raises(UnsupportedPlatform, match="no-replace rename"):
        publish(tmp_path / "result", lambda draft: None)

    assert list(tmp_path.iterdir()) == []


def test_macos_never_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    renameatx_np = FakeFunction(failing(errno.ENOTSUP))
    use_libc(monkeypatch, "darwin", FakeLibc(renameatx_np=renameatx_np))
    monkeypatch.setattr(os, "link", forbidden)

    with pytest.raises(UnsupportedPlatform, match="no-replace rename"):
        publish_file(tmp_path / "result", lambda path: path.write_text("x"))

    assert renameatx_np.calls[0][-1] == 0x4  # RENAME_EXCL


# -- fallback behaviour ----------------------------------------------------------------------
@pytest.mark.usefixtures("linux_fallback")
def test_regular_file_fallback_publishes_and_never_overwrites(tmp_path: Path) -> None:
    destination = tmp_path / "result.json"

    publication = publish_file(destination, lambda path: path.write_text("complete"))

    assert publication.cleanup_pending is False
    assert destination.read_text() == "complete"
    assert destination.stat().st_nlink == 1
    assert sorted(tmp_path.iterdir()) == [destination]
    with pytest.raises(DestinationExists, match="already exists"):
        publish_file(destination, lambda path: path.write_text("replacement"))
    assert destination.read_text() == "complete"
    assert sorted(tmp_path.iterdir()) == [destination]


@pytest.mark.usefixtures("linux_fallback")
def test_directory_fallback_publishes_complete_tree(tmp_path: Path) -> None:
    def build(draft: Draft) -> None:
        (draft.path / "nested").mkdir()
        (draft.path / "nested/value").write_text("complete")

    publication = publish(tmp_path / "result", build)

    assert publication.cleanup_pending is False
    assert (tmp_path / "result/nested/value").read_text() == "complete"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["result"]


@pytest.mark.usefixtures("linux_fallback")
def test_directory_fallback_rejects_group_writable_parent(tmp_path: Path) -> None:
    tmp_path.chmod(0o777)
    try:
        with pytest.raises(UnsafeFilesystem, match="owner-controlled"):
            publish(tmp_path / "result", lambda draft: None)
    finally:
        tmp_path.chmod(0o700)

    assert list(tmp_path.iterdir()) == []


def test_directory_fallback_rejects_source_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def substitute(*args: Any) -> int:
        source_fd, source = args[0], args[1]
        os.rename(source, "moved-stage", src_dir_fd=source_fd, dst_dir_fd=source_fd)
        os.mkdir(source, 0o700, dir_fd=source_fd)
        ctypes.set_errno(errno.EINVAL)
        return -1

    use_libc(monkeypatch, "linux", FakeLibc(renameat2=FakeFunction(substitute)))

    with pytest.raises(UnsafeFilesystem, match="substituted"):
        publish(tmp_path / "result", lambda draft: (draft.path / "value").write_text("ours"))

    assert not (tmp_path / "result").exists()
    assert (tmp_path / "moved-stage/value").read_text() == "ours"


@pytest.mark.usefixtures("linux_fallback")
def test_directory_fallback_releases_dedicated_lock_by_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "stage").mkdir()
    operations: list[tuple[int, int]] = []
    real_flock = fcntl.flock

    def record(fd: int, operation: int) -> None:
        operations.append((fd, operation))
        if operation == fcntl.LOCK_UN:
            raise OSError("explicit unlock must not run")
        real_flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", record)
    with _fs.open_path(tmp_path) as directory:
        _fs.commit(directory, "stage", directory, "result", directory.present("stage"))
        ((lock_fd, operation),) = operations
        assert operation == fcntl.LOCK_EX
        assert lock_fd != directory.fd
        with _fs.open_path(tmp_path) as other:
            real_flock(other.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # the lock was released

    assert (tmp_path / "result").is_dir()


@pytest.mark.usefixtures("linux_fallback")
def test_directory_fallback_fails_closed_without_flock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(fd: int, operation: int) -> None:
        raise OSError(errno.EOPNOTSUPP, "flock unsupported")

    monkeypatch.setattr(fcntl, "flock", unavailable)

    with pytest.raises(UnsupportedPlatform, match="lock is unavailable"):
        publish(tmp_path / "result", lambda draft: None)

    # The NFS lock file was tried too; it is never removed (that would split its holders).
    assert [path.name for path in tmp_path.iterdir()] == [".servatus.lock"]


@pytest.mark.usefixtures("linux_fallback")
def test_identity_fallback_installs_workspace_state(tmp_path: Path) -> None:
    with Workspace(tmp_path / "result", identity=b"request") as workspace:
        container = workspace.path.parent
        assert (container / ".identity").stat().st_nlink == 1
        assert list(container.glob(".identity-*.tmp")) == []
        (workspace.path / "value").write_text("x")
        workspace.publish(lambda draft: draft.link(workspace.path / "value", "value"))

    assert sorted(path.name for path in tmp_path.iterdir()) == ["result"]


@pytest.mark.usefixtures("linux_fallback")
def test_directory_fallback_is_absent_or_complete_for_readers(tmp_path: Path) -> None:
    source = tmp_path / "stage"
    source.mkdir()
    expected = {f"value-{index}" for index in range(100)}
    for name in expected:
        (source / name).write_text(name)
    destination = tmp_path / "result"
    seen_absent = threading.Event()
    seen_complete = threading.Event()
    stop = threading.Event()
    failures: list[set[str]] = []

    def read() -> None:
        while not stop.is_set():
            try:
                names = {entry.name for entry in destination.iterdir()}
            except FileNotFoundError:
                seen_absent.set()
            else:
                if names != expected:
                    failures.append(names)
                    return
                seen_complete.set()

    reader = threading.Thread(target=read)
    reader.start()
    try:
        assert seen_absent.wait(5)
        pinned_commit(tmp_path, "stage", "result")
        assert seen_complete.wait(5)
    finally:
        stop.set()
        reader.join(timeout=5)

    assert failures == []


# -- cross-process fallback ------------------------------------------------------------------
def _commit_worker(parent: str, source: str, start: Connection, result: Connection) -> None:
    force_fallback_in_this_process()
    try:
        with _fs.open_path(parent) as directory:
            expected = directory.present(source)
            start.recv()
            try:
                _fs.commit(directory, source, directory, "result", expected)
            except DestinationExists:
                result.send("exists")
            else:
                result.send("committed")
    except BaseException as error:
        result.send(type(error).__name__)
        raise


def test_directory_fallback_has_one_winner_across_processes(tmp_path: Path) -> None:
    for name in ("stage-a", "stage-b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "value").write_text(name)
    context = multiprocessing.get_context("spawn")
    workers: list[tuple[BaseProcess, Connection, Connection]] = []
    for source in ("stage-a", "stage-b"):
        start_child, start_parent = context.Pipe(duplex=False)
        result_parent, result_child = context.Pipe(duplex=False)
        process = context.Process(
            target=_commit_worker, args=(str(tmp_path), source, start_child, result_child)
        )
        process.start()
        start_child.close()
        result_child.close()
        workers.append((process, start_parent, result_parent))

    for _, start, _ in workers:
        start.send("start")
        start.close()
    outcomes: list[str] = []
    for process, _, result in workers:
        assert result.poll(10)
        outcomes.append(result.recv())
        process.join(timeout=10)
        assert process.exitcode == 0

    assert sorted(outcomes) == ["committed", "exists"]
    assert (tmp_path / "result/value").read_text() in {"stage-a", "stage-b"}


class _Gate(NamedTuple):
    ready: Connection
    release: Connection


def _publish_worker(
    parent: str,
    destination: str,
    value: str,
    build_gate: _Gate | None,
    sync_gate: _Gate | None,
    result: Connection,
) -> None:
    force_fallback_in_this_process()
    parent_entry = os.stat(parent)
    real_fsync = os.fsync
    blocked: list[bool] = []

    def controlled_fsync(fd: int) -> None:
        if (
            sync_gate is not None
            and not blocked
            and _fs.same(os.fstat(fd), parent_entry)
            and os.path.isdir(os.path.join(parent, destination))
        ):
            blocked.append(True)
            sync_gate.ready.send("sync")
            if not sync_gate.release.poll(10):
                raise AssertionError("publication sync was not released")
            sync_gate.release.recv()
        real_fsync(fd)

    def build(draft: Draft) -> None:
        (draft.path / "value").write_text(value)
        if build_gate is not None:
            build_gate.ready.send("built")
            if not build_gate.release.poll(10):
                raise AssertionError("publication builder was not released")
            build_gate.release.recv()

    os.fsync = controlled_fsync
    try:
        publish(os.path.join(parent, destination), build)
    except DestinationExists:
        result.send("exists")
    except BaseException as error:
        result.send(type(error).__name__)
        raise
    else:
        result.send("committed")


def _gate() -> tuple[_Gate, _Gate]:
    """Return (worker side, test side) of a ready/release pair."""
    context = multiprocessing.get_context("spawn")
    ready_test, ready_worker = context.Pipe(duplex=False)
    release_worker, release_test = context.Pipe(duplex=False)
    return _Gate(ready_worker, release_worker), _Gate(ready_test, release_test)


def _launch(*args: object) -> tuple[BaseProcess, Connection]:
    context = multiprocessing.get_context("spawn")
    result_test, result_worker = context.Pipe(duplex=False)
    process = context.Process(target=_publish_worker, args=(*args, result_worker))
    process.start()
    result_worker.close()
    return process, result_test


def test_directory_fallback_does_not_hold_parent_lock_across_durability_sync(
    tmp_path: Path,
) -> None:
    same_worker, same_test = _gate()
    same, same_result = _launch(str(tmp_path), "result-a", "challenger", same_worker, None)
    assert same_test.ready.poll(10)
    assert same_test.ready.recv() == "built"

    sync_worker, sync_test = _gate()
    winner, winner_result = _launch(str(tmp_path), "result-a", "winner", None, sync_worker)
    assert sync_test.ready.poll(10)
    assert sync_test.ready.recv() == "sync"

    other, other_result = _launch(str(tmp_path), "result-b", "other", None, None)
    same_test.release.send("continue")
    try:
        assert other_result.poll(10)
        assert other_result.recv() == "committed"
        assert same_result.poll(10)
        assert same_result.recv() == "exists"
        assert not winner_result.poll()
        assert (tmp_path / "result-a/value").read_text() == "winner"
        assert (tmp_path / "result-b/value").read_text() == "other"
    finally:
        sync_test.release.send("continue")
        for process in (same, other, winner):
            process.join(timeout=10)

    assert winner_result.poll(5)
    assert winner_result.recv() == "committed"
    assert all(process.exitcode == 0 for process in (same, other, winner))


@pytest.mark.usefixtures("linux_fallback")
def test_file_fallback_verifies_the_hard_linked_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_link = os.link

    def link_then_substitute(source: Any, destination: Any, **kwargs: Any) -> None:
        real_link(source, destination, **kwargs)
        os.unlink(destination, dir_fd=kwargs["dst_dir_fd"])
        (tmp_path / destination).write_text("impostor")

    monkeypatch.setattr(os, "link", link_then_substitute)

    with pytest.raises(UnsafeFilesystem, match="not the verified source"):
        publish_file(tmp_path / "result", lambda path: path.write_text("verified"))


NFS_LOCK_ERRORS = [errno.EBADF, errno.ENOLCK, errno.EOPNOTSUPP, errno.EINVAL]


def refuse_directory_locks(monkeypatch: pytest.MonkeyPatch, code: int) -> list[int]:
    """Behave like NFS: `flock` on a directory descriptor fails; regular files still lock."""
    real_flock = fcntl.flock
    locked_files: list[int] = []

    def flock(fd: int, operation: int) -> None:
        entry = os.fstat(fd)
        if stat.S_ISDIR(entry.st_mode):
            raise OSError(code, os.strerror(code))
        locked_files.append(entry.st_ino)
        real_flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", flock)
    return locked_files


@pytest.mark.usefixtures("linux_fallback")
@pytest.mark.parametrize("code", NFS_LOCK_ERRORS)
def test_directory_fallback_locks_a_file_where_directories_cannot_be_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    locked_files = refuse_directory_locks(monkeypatch, code)

    publication = publish(tmp_path / "result", lambda draft: (draft.path / "value").touch())

    lock = tmp_path / ".servatus.lock"
    assert publication.cleanup_pending is False
    assert sorted(path.name for path in tmp_path.iterdir()) == [".servatus.lock", "result"]
    assert stat.S_IMODE(lock.stat().st_mode) == 0o600
    assert locked_files == [lock.stat().st_ino]


@pytest.mark.usefixtures("linux_fallback")
def test_directory_fallback_lock_file_must_be_a_private_regular_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refuse_directory_locks(monkeypatch, errno.ENOLCK)
    (tmp_path / "elsewhere").write_text("x")
    (tmp_path / ".servatus.lock").symlink_to(tmp_path / "elsewhere")

    with pytest.raises(UnsafeFilesystem, match="lock .servatus.lock is not a plain file"):
        publish(tmp_path / "result", lambda draft: None)
    (tmp_path / ".servatus.lock").unlink()
    (tmp_path / ".servatus.lock").write_text("")
    (tmp_path / ".servatus.lock").chmod(0o644)
    with pytest.raises(UnsafeFilesystem, match="must be owner-only"):
        publish(tmp_path / "result", lambda draft: None)

    assert not (tmp_path / "result").exists()
