from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

from servatus import (
    CrossDevicePublication,
    DestinationExists,
    Draft,
    UnsafePublication,
    UnsupportedPlatform,
    WorkConflict,
    Workspace,
    WorkspaceBusy,
    _posix,
    _workspace,
    publish,
)


def hidden_entries(parent: Path) -> list[Path]:
    return sorted(parent.glob(".servatus-*"))


def test_publish_exposes_complete_directory_and_never_overwrites(tmp_path: Path) -> None:
    destination = tmp_path / "result"

    publication = publish(
        destination,
        lambda draft: (draft.path / "value.txt").write_text("first"),
    )

    assert publication.destination == destination
    assert publication.cleanup_pending is False
    assert (destination / "value.txt").read_text() == "first"
    assert hidden_entries(tmp_path) == []

    with pytest.raises(DestinationExists):
        publish(
            destination,
            lambda draft: (draft.path / "value.txt").write_text("second"),
        )

    assert (destination / "value.txt").read_text() == "first"
    assert hidden_entries(tmp_path) == []


def test_builder_failure_cleans_disposable_stage_and_propagates(tmp_path: Path) -> None:
    destination = tmp_path / "result"

    def fail(draft: Draft) -> None:
        (draft.path / "partial").write_text("not canonical")
        raise ValueError("invalid result")

    with pytest.raises(ValueError, match="invalid result"):
        publish(destination, fail)

    assert not destination.exists()
    assert hidden_entries(tmp_path) == []


def test_destination_created_during_build_wins_without_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "result"

    def race(draft: Draft) -> None:
        (draft.path / "ours").write_text("ours")
        destination.mkdir()
        (destination / "theirs").write_text("theirs")

    with pytest.raises(DestinationExists):
        publish(destination, race)

    assert (destination / "theirs").read_text() == "theirs"
    assert not (destination / "ours").exists()
    assert hidden_entries(tmp_path) == []


def test_workspace_preserves_failure_and_reopens_same_identity(tmp_path: Path) -> None:
    destination = tmp_path / "result"

    with Workspace(destination, identity=b"request-a") as workspace:
        checkpoint = workspace.path / "last.ckpt"
        checkpoint.write_bytes(b"checkpoint")

        def fail(draft: Draft) -> None:
            draft.link(checkpoint, "last.ckpt")
            raise RuntimeError("not complete")

        with pytest.raises(RuntimeError, match="not complete"):
            workspace.publish(fail)

    with Workspace(destination, identity=b"request-a") as workspace:
        assert (workspace.path / "last.ckpt").read_bytes() == b"checkpoint"
        publication = workspace.publish(
            lambda draft: draft.link(workspace.path / "last.ckpt", "last.ckpt")
        )

    assert publication.cleanup_pending is False
    assert (destination / "last.ckpt").read_bytes() == b"checkpoint"
    assert hidden_entries(tmp_path) == []


def test_workspace_rejects_different_identity(tmp_path: Path) -> None:
    destination = tmp_path / "result"
    with Workspace(destination, identity=b"request-a") as workspace:
        (workspace.path / "checkpoint").write_text("state")

    with pytest.raises(WorkConflict), Workspace(destination, identity=b"request-b"):
        pass


def test_workspace_lock_is_nonblocking(tmp_path: Path) -> None:
    destination = tmp_path / "result"
    with (
        Workspace(destination, identity=b"request"),
        pytest.raises(WorkspaceBusy),
        Workspace(destination, identity=b"request"),
    ):
        pass


def test_link_creates_regular_file_with_same_inode(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"value")
    destination = tmp_path / "result"

    publish(destination, lambda draft: draft.link(source, "nested/value.bin"))

    linked = destination / "nested/value.bin"
    assert linked.read_bytes() == b"value"
    assert linked.stat().st_ino == source.stat().st_ino


@pytest.mark.parametrize("unsafe", ["", ".", "..", "../escape", "/absolute"])
def test_link_rejects_escaping_paths(tmp_path: Path, unsafe: str) -> None:
    source = tmp_path / "source"
    source.write_text("value")

    with pytest.raises(UnsafePublication):
        publish(tmp_path / "result", lambda draft: draft.link(source, unsafe))

    assert not (tmp_path / "result").exists()


def test_link_rejects_occupied_draft_path(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_text("value")

    def build(draft: Draft) -> None:
        (draft.path / "value").write_text("occupied")
        draft.link(source, "value")

    with pytest.raises(DestinationExists):
        publish(tmp_path / "result", build)


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_publish_rejects_symlinks_and_special_files(tmp_path: Path, kind: str) -> None:
    def build(draft: Draft) -> None:
        if kind == "symlink":
            (draft.path / "unsafe").symlink_to(tmp_path / "outside")
        else:
            os.mkfifo(draft.path / "unsafe")

    with pytest.raises(UnsafePublication):
        publish(tmp_path / "result", build)

    assert not (tmp_path / "result").exists()
    assert hidden_entries(tmp_path) == []


def test_link_rejects_cross_device_source(tmp_path: Path) -> None:
    shared_memory = Path("/dev/shm")
    if not shared_memory.is_dir() or shared_memory.stat().st_dev == tmp_path.stat().st_dev:
        pytest.skip("no separate synthetic temporary filesystem is available")
    with tempfile.TemporaryDirectory(dir=shared_memory) as directory:
        source = Path(directory) / "source"
        source.write_text("value")
        with pytest.raises(CrossDevicePublication):
            publish(tmp_path / "result", lambda draft: draft.link(source, "source"))


def test_publish_rejects_stage_path_substitution(tmp_path: Path) -> None:
    moved_stage = tmp_path / "moved-stage"

    def substitute(draft: Draft) -> None:
        draft.path.rename(moved_stage)
        draft.path.symlink_to(moved_stage, target_is_directory=True)

    with pytest.raises(UnsafePublication):
        publish(tmp_path / "result", substitute)

    assert not (tmp_path / "result").exists()
    assert moved_stage.is_dir()


def test_workspace_rejects_parent_path_substitution(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    moved_parent = tmp_path / "moved-parent"
    destination = parent / "result"

    with Workspace(destination, identity=b"request") as workspace:
        parent.rename(moved_parent)
        parent.mkdir()
        with pytest.raises(UnsafePublication):
            workspace.publish(lambda draft: None)

    assert not destination.exists()
    assert not (moved_parent / "result").exists()


def test_syncs_files_before_directories_and_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[str, int]] = []
    real_fsync = _posix.sync_descriptor

    def record(descriptor: int) -> None:
        entry = os.fstat(descriptor)
        events.append(("directory" if entry.st_mode & 0o040000 else "file", entry.st_ino))
        real_fsync(descriptor)

    monkeypatch.setattr(_posix, "sync_descriptor", record)

    def build(draft: Draft) -> None:
        child = draft.path / "child"
        child.mkdir()
        (child / "value").write_text("value")

    publish(tmp_path / "result", build)

    file_index = next(index for index, event in enumerate(events) if event[0] == "file")
    assert all(event[0] == "directory" for event in events[file_index + 1 :])
    assert len(events[file_index + 1 :]) >= 3


def test_cleanup_failure_reports_committed_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "result"

    def fail_cleanup(parent_fd: int, name: str, expected: os.stat_result) -> None:
        del parent_fd, name, expected
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(_workspace, "_cleanup_workspace", fail_cleanup)

    with Workspace(destination, identity=b"request") as workspace:
        publication = workspace.publish(lambda draft: (draft.path / "value").write_text("complete"))

    assert publication.destination == destination
    assert publication.cleanup_pending is True
    assert (destination / "value").read_text() == "complete"
    assert len(hidden_entries(tmp_path)) == 1


def test_nested_workspace_can_feed_parent_publication(tmp_path: Path) -> None:
    destination = tmp_path / "study"
    with Workspace(destination, identity=b"study") as parent:
        trial_destination = parent.path / "trial-0"
        with Workspace(trial_destination, identity=b"trial") as trial:
            checkpoint = trial.path / "result.bin"
            checkpoint.write_bytes(b"trial result")
            trial.publish(lambda draft: draft.link(checkpoint, "result.bin"))

        parent.publish(
            lambda draft: draft.link(trial_destination / "result.bin", "trial-0/result.bin")
        )

    assert (destination / "trial-0/result.bin").read_bytes() == b"trial result"
    assert hidden_entries(tmp_path) == []


def test_unsupported_platform_fails_before_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = False

    def build(draft: Draft) -> None:
        nonlocal built
        built = True

    monkeypatch.setattr(sys, "platform", "win32")

    with pytest.raises(UnsupportedPlatform):
        publish(tmp_path / "result", build)
    assert built is False
