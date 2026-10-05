from __future__ import annotations

import fcntl
import os
import stat
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from servatus import _fs
from servatus.publication import Draft, Workspace, publish, publish_file

Sync = Callable[[int], None]


def key(entry: os.stat_result) -> tuple[int, int]:
    return entry.st_dev, entry.st_ino


def kind(fd: int) -> str:
    return "directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"


def test_directory_syncs_files_then_directories_then_parent(tmp_path: Path, syscalls: Any) -> None:
    events: list[tuple[str, tuple[int, int]]] = []

    def record(real: Sync, fd: int) -> None:
        events.append((kind(fd), key(os.fstat(fd))))
        real(fd)

    syscalls.on_sync(record)

    def build(draft: Draft) -> None:
        (draft.path / "child").mkdir()
        (draft.path / "child/value").write_text("value")

    publish(tmp_path / "result", build)

    result = tmp_path / "result"
    assert events == [
        ("file", key((result / "child/value").stat())),
        ("directory", key((result / "child").stat())),
        ("directory", key(result.stat())),
        ("directory", key(tmp_path.stat())),
    ]


def test_file_syncs_content_then_parent(tmp_path: Path, syscalls: Any) -> None:
    events: list[str] = []

    def record(real: Sync, fd: int) -> None:
        events.append(kind(fd))
        real(fd)

    syscalls.on_sync(record)

    publish_file(tmp_path / "result", lambda path: path.write_text("complete"))

    assert events == ["file", "directory"]


@pytest.mark.skipif(sys.platform != "darwin", reason="F_FULLFSYNC is macOS-only")
def test_macos_durability_uses_full_fsync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[int] = []
    real_fcntl: Any = fcntl.fcntl

    def record(fd: int, command: int, *args: Any) -> Any:
        commands.append(command)
        return real_fcntl(fd, command, *args)

    def plain_fsync(fd: int) -> None:
        pytest.fail("plain fsync used")

    monkeypatch.setattr(fcntl, "fcntl", record)
    monkeypatch.setattr(os, "fsync", plain_fsync)

    publish_file(tmp_path / "result", lambda path: path.write_text("complete"))

    assert commands.count(getattr(fcntl, "F_FULLFSYNC")) == 2  # noqa: B009


@pytest.mark.parametrize("platform", ["native", "fallback"])
def test_post_commit_parent_sync_failure_propagates_without_stage_residue(
    tmp_path: Path, syscalls: Any, request: pytest.FixtureRequest, platform: str
) -> None:
    if platform == "fallback":
        request.getfixturevalue("linux_fallback")
    destination = tmp_path / "result"
    parent = key(tmp_path.stat())

    def fail(real: Sync, fd: int) -> None:
        if key(os.fstat(fd)) == parent and destination.exists():
            raise OSError("injected post-commit parent sync failure")
        real(fd)

    syscalls.on_sync(fail)

    with pytest.raises(OSError, match="post-commit parent sync failure"):
        publish(destination, lambda draft: (draft.path / "value").write_text("complete"))
    with pytest.raises(OSError, match="post-commit parent sync failure"):
        publish_file(tmp_path / "file", lambda path: path.write_text("complete"))

    assert (destination / "value").read_text() == "complete"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["file", "result"]


@pytest.mark.usefixtures("linux_fallback")
def test_directory_fallback_syncs_parent_once_after_rename(tmp_path: Path, syscalls: Any) -> None:
    destination = tmp_path / "result"
    parent = key(tmp_path.stat())
    after_rename: list[int] = []

    def count(real: Sync, fd: int) -> None:
        if key(os.fstat(fd)) == parent and destination.is_dir():
            after_rename.append(fd)
        real(fd)

    syscalls.on_sync(count)

    publication = publish(destination, lambda draft: None)

    assert len(after_rename) == 1
    assert publication.cleanup_pending is False


def test_write_new_and_replace_file_sync_content_before_the_directory(
    tmp_path: Path, syscalls: Any
) -> None:
    events: list[str] = []

    def record(real: Sync, fd: int) -> None:
        events.append(kind(fd))
        real(fd)

    syscalls.on_sync(record)
    with _fs.open_path(tmp_path) as directory:
        _fs.write_new(directory.fd, "new", b"x")
        assert events == ["file"]
        _fs.replace_file(directory.fd, "state", b"y")

    assert events == ["file", "file", "directory"]


def test_replace_file_failure_before_rename_keeps_original(tmp_path: Path, syscalls: Any) -> None:
    (tmp_path / "state").write_bytes(b"original")

    def fail(real: Sync, fd: int) -> None:
        raise OSError("injected content sync failure")

    syscalls.on_sync(fail)
    with _fs.open_path(tmp_path) as directory, pytest.raises(OSError, match="content sync"):
        _fs.replace_file(directory.fd, "state", b"new")

    assert (tmp_path / "state").read_bytes() == b"original"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["state"]


# -- workspace initialization ----------------------------------------------------------------
def test_workspace_syncs_private_hierarchy_before_identity_exists(
    tmp_path: Path, syscalls: Any
) -> None:
    events: list[tuple[tuple[int, int], bool]] = []

    def record(real: Sync, fd: int) -> None:
        containers = list(tmp_path.glob(".servatus-*"))
        identity = bool(containers) and (containers[0] / ".identity").exists()
        events.append((key(os.fstat(fd)), identity))
        real(fd)

    syscalls.on_sync(record)
    with Workspace(tmp_path / "result", identity=b"request") as workspace:
        container = workspace.path.parent

    keys = [entry for entry, _ in events]
    before = [entry for entry, identity in events if not identity]
    lock, work = key((container / ".lock").stat()), key((container / "work").stat())
    container_key, parent = key(container.stat()), key(tmp_path.stat())
    assert {lock, work, container_key, parent} <= set(before)
    assert keys.index(container_key) > max(keys.index(lock), keys.index(work))
    assert keys.index(parent) > keys.index(container_key)
    assert events[-1] == (container_key, True)  # the identity commit itself is synced


@pytest.mark.parametrize("child", [False, True])
def test_workspace_initialization_syncs_parent_without_coordination(
    tmp_path: Path, syscalls: Any, child: bool
) -> None:
    root = Workspace(tmp_path / "result", identity=b"request")
    if child:
        with root:
            pass
    workspace = root.child("trial", identity=b"trial") if child else root
    parent = workspace.path.parent.parent
    parent_key = key(parent.stat())
    probed: list[bool] = []

    def check(real: Sync, fd: int) -> None:
        if key(os.fstat(fd)) == parent_key:
            with _fs.open_path(parent) as other:
                fcntl.flock(other.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # not held by Servatus
            probed.append(True)
        real(fd)

    syscalls.on_sync(check)
    with workspace:
        pass

    assert probed


@pytest.mark.parametrize("child", [False, True])
def test_parent_sync_failure_prevents_identity_and_allows_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, syscalls: Any, child: bool
) -> None:
    root = Workspace(tmp_path / "result", identity=b"request")
    if child:
        with root:
            pass
    workspace = root.child("trial", identity=b"trial") if child else root
    parent_key = key(workspace.path.parent.parent.stat())
    failing = [True]

    def fail(real: Sync, fd: int) -> None:
        if failing and key(os.fstat(fd)) == parent_key:
            raise OSError("injected parent sync failure")
        real(fd)

    syscalls.on_sync(fail)
    with pytest.raises(OSError, match="parent sync failure"), workspace:
        pytest.fail("undurable work was exposed")
    assert not (workspace.path.parent / ".identity").exists()

    failing.clear()
    with workspace:
        (workspace.path / "checkpoint").write_text("resumed")
    assert (workspace.path.parent / ".identity").is_file()


def test_workspace_reopen_does_not_sync_unchanged_container(tmp_path: Path, syscalls: Any) -> None:
    with Workspace(tmp_path / "result", identity=b"request") as workspace:
        container = key(workspace.path.parent.stat())
    synced: list[tuple[int, int]] = []

    def record(real: Sync, fd: int) -> None:
        synced.append(key(os.fstat(fd)))
        real(fd)

    syscalls.on_sync(record)
    with Workspace(tmp_path / "result", identity=b"request"):
        pass

    assert synced == []
    assert container not in synced


def test_identity_initialization_failure_allows_same_identity_resume(
    tmp_path: Path, syscalls: Any
) -> None:
    failed: list[bool] = []

    def fail_first(real: Sync, fd: int) -> None:
        if not failed:
            failed.append(True)
            raise OSError("injected identity sync failure")
        real(fd)

    syscalls.on_sync(fail_first)
    with (
        pytest.raises(OSError, match="identity sync failure"),
        Workspace(tmp_path / "result", identity=b"request"),
    ):
        pass

    with Workspace(tmp_path / "result", identity=b"request") as workspace:
        (workspace.path / "checkpoint").write_text("resumed")
        container = workspace.path.parent
    assert (container / ".identity").is_file()
    assert list(container.glob(".identity-*.tmp")) == []


def test_identity_commit_sync_failure_removes_identity_stage(tmp_path: Path, syscalls: Any) -> None:
    def fail_after_identity(real: Sync, fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode) and ".identity" in os.listdir(fd):
            raise OSError("injected identity commit sync failure")
        real(fd)

    syscalls.on_sync(fail_after_identity)
    with (
        pytest.raises(OSError, match="identity commit sync failure"),
        Workspace(tmp_path / "result", identity=b"request"),
    ):
        pass

    (container,) = tmp_path.glob(".servatus-*")
    assert list(container.glob(".identity-*")) == []
