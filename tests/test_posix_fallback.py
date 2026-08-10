from __future__ import annotations

import errno
import multiprocessing
import os
import stat
import sys
import threading
import warnings
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

from servatus import (
    CrossDevicePublication,
    DestinationExists,
    UnsafePublication,
    UnsupportedPlatform,
    Workspace,
    _posix,
    publish,
    publish_file,
)


def _force_linux_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")

    def unavailable(parent_fd: int, source: str, destination: str) -> None:
        del parent_fd, source, destination
        raise _posix._NoreplaceUnavailable

    monkeypatch.setattr(_posix, "_linux_rename_noreplace", unavailable)


def _directory_fallback_worker(
    parent: str,
    source: str,
    destination: str,
    start: Connection,
    result: Connection,
) -> None:
    parent_fd = _posix.open_directory(Path(parent))
    try:
        expected_parent = os.fstat(parent_fd)
        expected_source = os.stat(source, dir_fd=parent_fd, follow_symlinks=False)
        start.recv()
        try:
            _posix._locked_directory_noreplace(
                parent_fd,
                source,
                destination,
                expected_parent,
                expected_source,
            )
        except DestinationExists:
            result.send("exists")
        else:
            result.send("committed")
    except BaseException as error:
        result.send(type(error).__name__)
        raise
    finally:
        os.close(parent_fd)


def test_regular_file_fallback_publishes_and_never_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_linux_fallback(monkeypatch)
    destination = tmp_path / "result.json"

    publication = publish_file(destination, lambda stage: stage.write_text("complete"))

    assert publication.destination == destination
    assert publication.cleanup_pending is False
    assert destination.read_text() == "complete"
    assert list(tmp_path.glob(".servatus-file-stage-*")) == []

    with pytest.raises(DestinationExists):
        publish_file(destination, lambda stage: stage.write_text("replacement"))

    assert destination.read_text() == "complete"


def test_directory_fallback_publishes_complete_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_linux_fallback(monkeypatch)

    publication = publish(
        tmp_path / "result", lambda draft: (draft.path / "value").write_text("complete")
    )

    assert publication.cleanup_pending is False
    assert (publication.destination / "value").read_text() == "complete"


def test_identity_fallback_installs_workspace_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_linux_fallback(monkeypatch)

    with Workspace(tmp_path / "result", identity=b"request") as workspace:
        container = workspace.path.parent
        assert (container / ".identity").is_file()
        assert list(container.glob(".identity-*.tmp")) == []


def test_identity_fallback_reports_cleanup_residue_without_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_linux_fallback(monkeypatch)
    real_remove = _posix.remove_file_at

    def fail_identity_stage_cleanup(parent_fd: int, name: str, expected: os.stat_result) -> None:
        if name.startswith(".identity-"):
            raise OSError("injected identity stage cleanup failure")
        real_remove(parent_fd, name, expected)

    monkeypatch.setattr(_posix, "remove_file_at", fail_identity_stage_cleanup)

    with (
        pytest.warns(RuntimeWarning, match="identity-stage cleanup"),
        Workspace(tmp_path / "result", identity=b"request") as workspace,
    ):
        container = workspace.path.parent
        assert (container / ".identity").is_file()
        assert len(list(container.glob(".identity-*.tmp"))) == 1


def test_identity_warning_preserves_other_thread_warning_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_linux_fallback(monkeypatch)
    real_remove = _posix.remove_file_at
    real_warn = warnings.warn
    warning_started = threading.Event()
    continue_warning = threading.Event()
    failures: list[BaseException] = []

    def fail_identity_stage_cleanup(parent_fd: int, name: str, expected: os.stat_result) -> None:
        if name.startswith(".identity-"):
            raise OSError("injected identity stage cleanup failure")
        real_remove(parent_fd, name, expected)

    def pause_identity_warning(
        message: str | Warning,
        category: type[Warning] | None = None,
        stacklevel: int = 1,
        source: object | None = None,
    ) -> None:
        if "identity-stage cleanup" in str(message):
            warning_started.set()
            if not continue_warning.wait(5):
                raise AssertionError("identity warning did not resume")
        real_warn(message, category, stacklevel, source)

    def initialize() -> None:
        try:
            with Workspace(tmp_path / "result", identity=b"request"):
                pass
        except BaseException as error:
            failures.append(error)

    monkeypatch.setattr(_posix, "remove_file_at", fail_identity_stage_cleanup)
    monkeypatch.setattr(warnings, "warn", pause_identity_warning)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        worker = threading.Thread(target=initialize)
        worker.start()
        try:
            assert warning_started.wait(5)
            with pytest.raises(RuntimeWarning, match="unrelated"):
                real_warn("unrelated", RuntimeWarning)
        finally:
            continue_warning.set()
            worker.join(timeout=5)

    assert not worker.is_alive()
    assert failures == []


def test_identity_fallback_reconciles_unlinked_stage_without_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_linux_fallback(monkeypatch)
    real_sync = _posix.sync_descriptor
    failed_cleanup_sync = False

    def fail_first_sync_after_unlink(descriptor: int) -> None:
        nonlocal failed_cleanup_sync
        entry = os.fstat(descriptor)
        if stat.S_ISDIR(entry.st_mode):
            names = set(os.listdir(descriptor))
            identity_stage_absent = not any(name.startswith(".identity-") for name in names)
            if not failed_cleanup_sync and ".identity" in names and identity_stage_absent:
                failed_cleanup_sync = True
                raise OSError("injected identity cleanup sync failure")
        real_sync(descriptor)

    monkeypatch.setattr(_posix, "sync_descriptor", fail_first_sync_after_unlink)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with Workspace(tmp_path / "result", identity=b"request") as workspace:
            container = workspace.path.parent
            assert (container / ".identity").is_file()
            assert list(container.glob(".identity-*.tmp")) == []

    assert failed_cleanup_sync is True
    assert caught == []


def test_identity_fallback_warns_when_cleanup_sync_retry_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_linux_fallback(monkeypatch)
    real_sync = _posix.sync_descriptor
    failed_cleanup_syncs = 0

    def fail_cleanup_syncs(descriptor: int) -> None:
        nonlocal failed_cleanup_syncs
        entry = os.fstat(descriptor)
        if stat.S_ISDIR(entry.st_mode):
            names = set(os.listdir(descriptor))
            if ".identity" in names and not any(name.startswith(".identity-") for name in names):
                failed_cleanup_syncs += 1
                raise OSError("injected identity cleanup sync failure")
        real_sync(descriptor)

    monkeypatch.setattr(_posix, "sync_descriptor", fail_cleanup_syncs)

    with (
        pytest.warns(RuntimeWarning, match="identity-stage cleanup"),
        Workspace(tmp_path / "result", identity=b"request") as workspace,
    ):
        container = workspace.path.parent
        assert (container / ".identity").is_file()
        assert list(container.glob(".identity-*.tmp")) == []

    assert failed_cleanup_syncs == 2


def test_file_fallback_reports_only_private_cleanup_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_linux_fallback(monkeypatch)

    def fail_cleanup(parent_fd: int, name: str, expected: os.stat_result) -> None:
        del parent_fd, name, expected
        raise OSError("injected stage cleanup failure")

    monkeypatch.setattr(_posix, "remove_file_at", fail_cleanup)

    publication = publish_file(tmp_path / "result", lambda stage: stage.write_text("complete"))

    assert publication.cleanup_pending is True
    assert publication.destination.read_text() == "complete"
    assert len(list(tmp_path.glob(".servatus-file-stage-*"))) == 1


def test_directory_fallback_has_one_winner_across_processes(tmp_path: Path) -> None:
    for name in ("stage-a", "stage-b"):
        stage = tmp_path / name
        stage.mkdir()
        (stage / "value").write_text(name)

    context = multiprocessing.get_context("spawn")
    processes: list[multiprocessing.Process] = []
    starts: list[Connection] = []
    results: list[Connection] = []
    for source in ("stage-a", "stage-b"):
        start_child, start_parent = context.Pipe(duplex=False)
        result_parent, result_child = context.Pipe(duplex=False)
        process = context.Process(
            target=_directory_fallback_worker,
            args=(str(tmp_path), source, "result", start_child, result_child),
        )
        process.start()
        start_child.close()
        result_child.close()
        processes.append(process)
        starts.append(start_parent)
        results.append(result_parent)

    for start in starts:
        start.send("start")
        start.close()
    outcomes = []
    for result in results:
        assert result.poll(10)
        outcomes.append(result.recv())
        result.close()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert sorted(outcomes) == ["committed", "exists"]
    assert (tmp_path / "result/value").read_text() in {"stage-a", "stage-b"}


def test_directory_fallback_is_absent_or_complete_for_readers(tmp_path: Path) -> None:
    source = tmp_path / "stage"
    source.mkdir()
    expected_names = {f"value-{index}" for index in range(100)}
    for name in expected_names:
        (source / name).write_text(name)

    seen_absent = threading.Event()
    seen_complete = threading.Event()
    stop = threading.Event()
    failures: list[set[str]] = []
    destination = tmp_path / "result"

    def read() -> None:
        while not stop.is_set():
            try:
                names = {entry.name for entry in destination.iterdir()}
            except FileNotFoundError:
                seen_absent.set()
            else:
                if names == expected_names:
                    seen_complete.set()
                else:
                    failures.append(names)
                    return

    reader = threading.Thread(target=read)
    reader.start()
    assert seen_absent.wait(5)
    parent_fd = _posix.open_directory(tmp_path)
    try:
        _posix._locked_directory_noreplace(
            parent_fd,
            "stage",
            "result",
            os.fstat(parent_fd),
            source.stat(follow_symlinks=False),
        )
        assert seen_complete.wait(5)
    finally:
        stop.set()
        reader.join(timeout=5)
        os.close(parent_fd)

    assert failures == []


def test_directory_fallback_fails_closed_when_flock_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_linux_fallback(monkeypatch)

    def unavailable(descriptor: int, operation: int) -> None:
        del descriptor, operation
        raise OSError(errno.EOPNOTSUPP, "flock unsupported")

    monkeypatch.setattr(_posix.fcntl, "flock", unavailable)

    with pytest.raises(UnsupportedPlatform, match="lock"):
        publish(tmp_path / "result", lambda draft: None)

    assert not (tmp_path / "result").exists()


def test_directory_fallback_releases_dedicated_lock_by_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "stage"
    source.mkdir()
    parent_fd = _posix.open_directory(tmp_path)
    operations: list[int] = []
    lock_descriptors: list[int] = []
    real_flock = _posix.fcntl.flock

    def reject_explicit_unlock(descriptor: int, operation: int) -> None:
        lock_descriptors.append(descriptor)
        operations.append(operation)
        if operation == _posix.fcntl.LOCK_UN:
            raise OSError("explicit unlock must not run after commit")
        real_flock(descriptor, operation)

    monkeypatch.setattr(_posix.fcntl, "flock", reject_explicit_unlock)
    try:
        _posix._locked_directory_noreplace(
            parent_fd,
            "stage",
            "result",
            os.fstat(parent_fd),
            source.stat(follow_symlinks=False),
        )
        os.fstat(parent_fd)
    finally:
        os.close(parent_fd)

    assert operations == [_posix.fcntl.LOCK_EX]
    assert lock_descriptors[0] != parent_fd
    assert (tmp_path / "result").is_dir()


def test_directory_fallback_rejects_source_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")

    def substitute(parent_fd: int, source: str, destination: str) -> None:
        del destination
        os.rename(source, "moved-stage", src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        os.mkdir(source, dir_fd=parent_fd)
        raise _posix._NoreplaceUnavailable

    monkeypatch.setattr(_posix, "_linux_rename_noreplace", substitute)

    with pytest.raises(UnsafePublication):
        publish(tmp_path / "result", lambda draft: (draft.path / "value").write_text("ours"))

    assert not (tmp_path / "result").exists()
    assert (tmp_path / "moved-stage/value").read_text() == "ours"


def test_fallback_rejects_parent_descriptor_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    original = tmp_path / "original"
    replacement = tmp_path / "replacement"
    original.mkdir()
    replacement.mkdir()
    source = original / "stage"
    source.write_text("complete")
    os.link(source, replacement / "stage")
    parent_fd = _posix.open_directory(original)

    def substitute(parent_fd: int, source: str, destination: str) -> None:
        del source, destination
        replacement_fd = _posix.open_directory(replacement)
        try:
            os.dup2(replacement_fd, parent_fd)
        finally:
            os.close(replacement_fd)
        raise _posix._NoreplaceUnavailable

    monkeypatch.setattr(_posix, "_linux_rename_noreplace", substitute)
    try:
        with pytest.raises(UnsafePublication, match="parent"):
            _posix.commit_noreplace(
                parent_fd,
                "stage",
                "result",
                source.stat(follow_symlinks=False),
            )
    finally:
        os.close(parent_fd)

    assert not (original / "result").exists()
    assert not (replacement / "result").exists()


def test_native_noreplace_remains_the_fast_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    source = tmp_path / "stage"
    source.mkdir()
    parent_fd = _posix.open_directory(tmp_path)
    synced = False

    def native(parent_fd: int, source: str, destination: str) -> None:
        os.rename(source, destination, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)

    def unexpected(*args: object) -> None:
        del args
        raise AssertionError("fallback must not run")

    real_sync = _posix.sync_descriptor

    def record_sync(descriptor: int) -> None:
        nonlocal synced
        synced = True
        real_sync(descriptor)

    monkeypatch.setattr(_posix, "_linux_rename_noreplace", native)
    monkeypatch.setattr(_posix, "_locked_directory_noreplace", unexpected)
    monkeypatch.setattr(_posix, "sync_descriptor", record_sync)
    try:
        outcome = _posix.commit_noreplace(
            parent_fd,
            "stage",
            "result",
            source.stat(follow_symlinks=False),
        )
    finally:
        os.close(parent_fd)

    assert outcome.cleanup_pending is False
    assert synced is True
    assert (tmp_path / "result").is_dir()


def test_cross_device_and_unexpected_native_errors_do_not_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    source = tmp_path / "stage"
    source.mkdir()
    parent_fd = _posix.open_directory(tmp_path)

    def unexpected_fallback(*args: object) -> None:
        del args
        raise AssertionError("fallback must not run")

    monkeypatch.setattr(_posix, "_locked_directory_noreplace", unexpected_fallback)
    try:
        monkeypatch.setattr(
            _posix,
            "_linux_rename_noreplace",
            lambda *args: (_ for _ in ()).throw(
                CrossDevicePublication("publication crosses a filesystem")
            ),
        )
        with pytest.raises(CrossDevicePublication):
            _posix.commit_noreplace(
                parent_fd,
                "stage",
                "result",
                source.stat(follow_symlinks=False),
            )

        failure = OSError(errno.EIO, "injected I/O failure")
        monkeypatch.setattr(
            _posix,
            "_linux_rename_noreplace",
            lambda *args: (_ for _ in ()).throw(failure),
        )
        with pytest.raises(OSError) as raised:
            _posix.commit_noreplace(
                parent_fd,
                "stage",
                "result",
                source.stat(follow_symlinks=False),
            )
        assert raised.value is failure
    finally:
        os.close(parent_fd)


def test_directory_fallback_rejects_unsafe_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_linux_fallback(monkeypatch)
    tmp_path.chmod(0o777)
    try:
        with pytest.raises(UnsafePublication, match="owner-controlled"):
            publish(tmp_path / "result", lambda draft: None)
    finally:
        tmp_path.chmod(0o700)
