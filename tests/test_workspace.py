from __future__ import annotations

import hashlib
import os
import stat
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
    publish_file,
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


def test_publish_file_exposes_complete_regular_file_with_ordinary_writer_mode(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "result.json"
    observed: tuple[int, int, int] | None = None

    def write(stage: Path) -> None:
        nonlocal observed
        entry = stage.stat(follow_symlinks=False)
        observed = (entry.st_size, entry.st_dev, entry.st_ino)
        stage.write_text('{"status":"complete"}\n')

    previous_umask = os.umask(0o022)
    try:
        publication = publish_file(destination, write)
    finally:
        os.umask(previous_umask)

    assert observed is not None
    assert observed[0] == 0
    assert observed[1] == tmp_path.stat().st_dev
    assert publication.destination == destination
    assert publication.cleanup_pending is False
    assert destination.read_text() == '{"status":"complete"}\n'
    assert destination.stat().st_ino == observed[2]
    assert destination.stat().st_mode & 0o777 == 0o644
    assert hidden_entries(tmp_path) == []


def test_publish_file_supports_binary_writers_and_preserves_explicit_mode(tmp_path: Path) -> None:
    destination = tmp_path / "result.parquet"

    def write(stage: Path) -> None:
        with stage.open("wb") as stream:
            stream.write(b"PAR1\x00payload")
        stage.chmod(0o640)

    publish_file(destination, write)

    assert destination.read_bytes() == b"PAR1\x00payload"
    assert destination.stat().st_mode & 0o777 == 0o640


def test_publish_file_writer_failure_cleans_stage_and_propagates_same_exception(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "result"
    failure = ValueError("invalid result")

    def fail(stage: Path) -> None:
        stage.write_text("not canonical")
        raise failure

    with pytest.raises(ValueError) as raised:
        publish_file(destination, fail)

    assert raised.value is failure
    assert not destination.exists()
    assert hidden_entries(tmp_path) == []


@pytest.mark.parametrize("replacement", ["missing", "regular", "directory", "symlink"])
def test_publish_file_rejects_stage_path_substitution(tmp_path: Path, replacement: str) -> None:
    destination = tmp_path / "result"
    symlink_target = tmp_path / "target"
    symlink_target.write_text("target")

    def substitute(stage: Path) -> None:
        stage.unlink()
        if replacement == "regular":
            stage.write_text("replacement")
        elif replacement == "directory":
            stage.mkdir()
        elif replacement == "symlink":
            stage.symlink_to(symlink_target)

    with pytest.raises(UnsafePublication):
        publish_file(destination, substitute)

    assert not destination.exists()
    replacements = hidden_entries(tmp_path)
    if replacement == "missing":
        assert replacements == []
    else:
        assert len(replacements) == 1


def test_publish_file_destination_race_never_overwrites(tmp_path: Path) -> None:
    destination = tmp_path / "result"

    def race(stage: Path) -> None:
        stage.write_text("ours")
        destination.write_text("theirs")

    with pytest.raises(DestinationExists):
        publish_file(destination, race)

    assert destination.read_text() == "theirs"
    assert hidden_entries(tmp_path) == []


def test_publish_file_syncs_file_before_committed_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    real_sync = _posix.sync_descriptor

    def record(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        events.append("file" if stat.S_ISREG(mode) else "directory")
        real_sync(descriptor)

    monkeypatch.setattr(_posix, "sync_descriptor", record)

    publish_file(tmp_path / "result", lambda stage: stage.write_text("complete"))

    assert events == ["file", "directory"]


def test_publish_file_cleanup_failure_does_not_replace_writer_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = RuntimeError("invalid output")

    def fail_cleanup(parent_fd: int, name: str, expected: os.stat_result) -> None:
        del parent_fd, name, expected
        raise OSError("injected cleanup failure")

    def fail(stage: Path) -> None:
        stage.write_text("partial")
        raise failure

    monkeypatch.setattr(_posix, "remove_file_at", fail_cleanup)

    with pytest.raises(RuntimeError) as raised:
        publish_file(tmp_path / "result", fail)

    assert raised.value is failure
    assert any("injected cleanup failure" in note for note in failure.__notes__)


def test_failed_attempt_closes_stage_descriptor_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_fd = -1
    stage_closes = 0
    real_make_stage = _posix.make_unique_stage
    real_close = os.close

    def track_stage(parent_fd: int, prefix: str) -> tuple[str, int, os.stat_result]:
        nonlocal stage_fd
        result = real_make_stage(parent_fd, prefix)
        stage_fd = result[1]
        return result

    def track_close(descriptor: int) -> None:
        nonlocal stage_closes
        if descriptor == stage_fd:
            stage_closes += 1
        real_close(descriptor)

    monkeypatch.setattr(_posix, "make_unique_stage", track_stage)
    monkeypatch.setattr(_posix.os, "close", track_close)

    def fail(draft: Draft) -> None:
        raise ValueError("stop")

    with pytest.raises(ValueError, match="stop"):
        publish(tmp_path / "result", fail)

    assert stage_closes == 1


def test_publish_rejects_new_stage_substitution_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "result"
    moved_stage = tmp_path / "moved-stage"
    replacement_stage: Path | None = None
    substituted = False
    real_open_directory_at = _posix.open_directory_at

    def substitute(parent_fd: int, name: str) -> int:
        nonlocal replacement_stage, substituted
        if not substituted and name.startswith(".servatus-stage-"):
            replacement_stage = tmp_path / name
            replacement_stage.rename(moved_stage)
            replacement_stage.mkdir(mode=0o700)
            (replacement_stage / "injected").write_text("preserve")
            substituted = True
        return real_open_directory_at(parent_fd, name)

    monkeypatch.setattr(_posix, "open_directory_at", substitute)

    with pytest.raises(UnsafePublication):
        publish(destination, lambda draft: None)

    assert substituted is True
    assert not destination.exists()
    assert moved_stage.is_dir()
    assert replacement_stage is not None
    assert (replacement_stage / "injected").read_text() == "preserve"


def test_publish_rejects_nul_destination_before_build(tmp_path: Path) -> None:
    built = False

    def build(draft: Draft) -> None:
        nonlocal built
        built = True

    with pytest.raises(UnsafePublication):
        publish(tmp_path / "result\0truncated", build)

    assert built is False
    assert not (tmp_path / "result").exists()
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


@pytest.mark.parametrize(
    ("entry_name", "mode"),
    [
        ("container", 0o750),
        ("work", 0o750),
        ("lock", 0o640),
        ("identity", 0o640),
    ],
)
def test_workspace_rejects_permissive_private_entries(
    tmp_path: Path, entry_name: str, mode: int
) -> None:
    destination = tmp_path / "result"
    with Workspace(destination, identity=b"request"):
        pass
    container = hidden_entries(tmp_path)[0]
    entries = {
        "container": container,
        "work": container / "work",
        "lock": container / ".lock",
        "identity": container / ".identity",
    }
    entries[entry_name].chmod(mode)

    with pytest.raises(UnsafePublication), Workspace(destination, identity=b"request"):
        pass


@pytest.mark.parametrize(
    ("entry_name", "mode"),
    [("container", 0o770), ("work", 0o770), ("lock", 0o660)],
)
def test_workspace_rejects_permissions_changed_after_enter(
    tmp_path: Path, entry_name: str, mode: int
) -> None:
    destination = tmp_path / "result"
    with Workspace(destination, identity=b"request") as workspace:
        container = workspace.path.parent
        entries = {
            "container": container,
            "work": workspace.path,
            "lock": container / ".lock",
        }
        entries[entry_name].chmod(mode)

        with pytest.raises(UnsafePublication):
            workspace.publish(lambda draft: None)

    assert not destination.exists()


@pytest.mark.parametrize("entry_name", ["container", "work", "lock", "identity"])
def test_workspace_rejects_foreign_private_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry_name: str
) -> None:
    destination = tmp_path / "result"
    with Workspace(destination, identity=b"request"):
        pass
    container = hidden_entries(tmp_path)[0]
    entries = {
        "container": container,
        "work": container / "work",
        "lock": container / ".lock",
        "identity": container / ".identity",
    }
    target_inode = entries[entry_name].stat(follow_symlinks=False).st_ino
    real_fstat = os.fstat

    def report_foreign_owner(descriptor: int) -> os.stat_result:
        entry = real_fstat(descriptor)
        if entry.st_ino != target_inode:
            return entry
        fields = list(entry)
        fields[4] = entry.st_uid + 1
        return os.stat_result(fields)

    monkeypatch.setattr(_workspace.os, "fstat", report_foreign_owner)

    with pytest.raises(UnsafePublication), Workspace(destination, identity=b"request"):
        pass


def test_workspace_identity_stores_canonical_inode_pins(tmp_path: Path) -> None:
    destination = tmp_path / "result"
    with Workspace(destination, identity=b"request"):
        pass
    container = hidden_entries(tmp_path)[0]
    entries = (container, container / ".lock", container / "work")
    expected = (
        b"servatus-workspace-v2\n"
        + hashlib.sha256(b"request").hexdigest().encode("ascii")
        + b"\n"
        + b"".join(f"{entry.stat(follow_symlinks=False).st_ino}\n".encode() for entry in entries)
    )

    assert (container / ".identity").read_bytes() == expected


@pytest.mark.parametrize("pin_index", [0, 1, 2])
def test_workspace_rejects_changed_persisted_inode(tmp_path: Path, pin_index: int) -> None:
    destination = tmp_path / "result"
    with Workspace(destination, identity=b"request"):
        pass
    identity_path = hidden_entries(tmp_path)[0] / ".identity"
    header, digest, *pins = identity_path.read_bytes().splitlines()
    pins[pin_index] = str(int(pins[pin_index]) + 1).encode("ascii")
    identity_path.write_bytes(b"\n".join((header, digest, *pins)) + b"\n")

    with pytest.raises(UnsafePublication), Workspace(destination, identity=b"request"):
        pass


@pytest.mark.parametrize(
    "pins",
    [
        b"2\n4\n",
        b"inode\n4\n6\n",
        b"02\n4\n6\n",
        b"18446744073709551616\n4\n6\n",
        b"2\n4\n6\nextra\n",
    ],
)
def test_workspace_rejects_malformed_persisted_pins(tmp_path: Path, pins: bytes) -> None:
    destination = tmp_path / "result"
    with Workspace(destination, identity=b"request"):
        pass
    identity_path = hidden_entries(tmp_path)[0] / ".identity"
    header, digest, *_ = identity_path.read_bytes().splitlines(keepends=True)
    identity_path.write_bytes(header + digest + pins)

    with pytest.raises(UnsafePublication), Workspace(destination, identity=b"request"):
        pass


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


@pytest.mark.parametrize("unsafe", ["", ".", "..", "../escape", "/absolute", "safe\0truncated"])
def test_link_rejects_escaping_paths(tmp_path: Path, unsafe: str) -> None:
    source = tmp_path / "source"
    source.write_text("value")

    with pytest.raises(UnsafePublication):
        publish(tmp_path / "result", lambda draft: draft.link(source, unsafe))

    assert not (tmp_path / "result").exists()


def test_link_rejects_nul_source_path(tmp_path: Path) -> None:
    with pytest.raises(UnsafePublication):
        publish(
            tmp_path / "result",
            lambda draft: draft.link(tmp_path / "source\0truncated", "source"),
        )

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


def test_workspace_rejects_new_directory_substitution_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    moved_container = tmp_path / "moved-container"
    replacement_container: Path | None = None
    substituted = False
    real_open_directory_at = _posix.open_directory_at

    def substitute(parent_fd: int, name: str) -> int:
        nonlocal replacement_container, substituted
        if not substituted and name.startswith(".servatus-") and name.endswith(".work"):
            replacement_container = tmp_path / name
            replacement_container.rename(moved_container)
            replacement_container.mkdir(mode=0o700)
            substituted = True
        return real_open_directory_at(parent_fd, name)

    monkeypatch.setattr(_posix, "open_directory_at", substitute)

    with pytest.raises(UnsafePublication), Workspace(tmp_path / "result", identity=b"request"):
        pass

    assert substituted is True
    assert moved_container.is_dir()
    assert replacement_container is not None
    assert replacement_container.is_dir()


def test_parent_substitution_during_build_cannot_redirect_commit(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    moved_parent = tmp_path / "moved-parent"
    destination = parent / "result"
    replacement_stage: Path | None = None

    def substitute(draft: Draft) -> None:
        nonlocal replacement_stage
        (draft.path / "ours").write_text("ours")
        stage_name = draft.path.name
        parent.rename(moved_parent)
        parent.mkdir()
        replacement_stage = parent / stage_name
        replacement_stage.mkdir()
        (replacement_stage / "theirs").write_text("theirs")

    with pytest.raises(UnsafePublication):
        publish(destination, substitute)

    assert not destination.exists()
    assert not (moved_parent / "result").exists()
    assert replacement_stage is not None
    assert (replacement_stage / "theirs").read_text() == "theirs"


def test_descriptor_bound_commit_resists_last_moment_parent_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    moved_parent = tmp_path / "moved-parent"
    replacement_parent = tmp_path / "replacement-parent"
    destination = parent / "result"
    real_commit = _posix.commit_noreplace

    def substitute(
        parent_fd: int,
        source: str,
        destination_name: str,
        expected_source: os.stat_result,
    ) -> _posix._CommitOutcome:
        parent.rename(moved_parent)
        parent.mkdir()
        colliding_stage = parent / source
        colliding_stage.mkdir()
        (colliding_stage / "theirs").write_text("theirs")
        try:
            return real_commit(parent_fd, source, destination_name, expected_source)
        finally:
            parent.rename(replacement_parent)
            moved_parent.rename(parent)

    monkeypatch.setattr(_posix, "commit_noreplace", substitute)

    publication = publish(
        destination,
        lambda draft: (draft.path / "ours").write_text("ours"),
    )

    assert (publication.destination / "ours").read_text() == "ours"
    colliding_stage = next(replacement_parent.glob(".servatus-stage-*"))
    assert (colliding_stage / "theirs").read_text() == "theirs"
    assert not (replacement_parent / "result").exists()


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

    assert [kind for kind, _ in events] == [
        "file",
        "directory",
        "directory",
        "directory",
    ]


def test_cleanup_failure_reports_committed_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "result"

    def fail_cleanup(parent_fd: int, name: str, expected: os.stat_result) -> None:
        del parent_fd, name, expected
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(_posix, "remove_tree_at", fail_cleanup)

    with Workspace(destination, identity=b"request") as workspace:
        publication = workspace.publish(lambda draft: (draft.path / "value").write_text("complete"))

    assert publication.destination == destination
    assert publication.cleanup_pending is True
    assert (destination / "value").read_text() == "complete"
    assert len(hidden_entries(tmp_path)) == 1


def test_workspace_cleanup_preserves_substituted_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "result"
    moved_container = tmp_path / "moved-container"
    with Workspace(destination, identity=b"request") as workspace:
        container = workspace.path.parent
        container_checks = 0
        real_ensure_entry = _posix.ensure_entry

        def substitute_after_cleanup_check(
            parent_fd: int, name: str, expected: os.stat_result
        ) -> None:
            nonlocal container_checks
            real_ensure_entry(parent_fd, name, expected)
            if name != container.name:
                return
            container_checks += 1
            if container_checks == 3:
                container.rename(moved_container)
                container.mkdir(mode=0o700)
                (container / "replacement").write_text("preserve")

        monkeypatch.setattr(_posix, "ensure_entry", substitute_after_cleanup_check)
        publication = workspace.publish(lambda draft: None)

    assert publication.cleanup_pending is True
    assert moved_container.is_dir()
    assert (container / "replacement").read_text() == "preserve"


def test_workspace_cleanup_preserves_substituted_nested_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "result"
    moved_nested = tmp_path / "moved-nested"
    substituted = False
    with Workspace(destination, identity=b"request") as workspace:
        nested = workspace.path / "nested"
        nested.mkdir()
        (nested / "checkpoint").write_text("preserve")
        real_ensure_entry = _posix.ensure_entry

        def substitute_after_cleanup_check(
            parent_fd: int, name: str, expected: os.stat_result
        ) -> None:
            nonlocal substituted
            real_ensure_entry(parent_fd, name, expected)
            if name == nested.name and not substituted:
                nested.rename(moved_nested)
                nested.mkdir(mode=0o700)
                (nested / "replacement").write_text("preserve")
                substituted = True

        monkeypatch.setattr(_posix, "ensure_entry", substitute_after_cleanup_check)
        publication = workspace.publish(lambda draft: None)

    assert publication.cleanup_pending is True
    assert substituted is True
    assert moved_nested.is_dir()
    assert (nested / "replacement").read_text() == "preserve"


def test_workspace_cleanup_removes_unreadable_regular_file(tmp_path: Path) -> None:
    destination = tmp_path / "result"
    with Workspace(destination, identity=b"request") as workspace:
        private_file = workspace.path / "private"
        private_file.write_text("resumable")
        private_file.chmod(0o000)
        publication = workspace.publish(lambda draft: None)

    assert publication.cleanup_pending is False
    assert destination.is_dir()
    assert hidden_entries(tmp_path) == []


def test_workspace_syncs_new_private_hierarchy_before_identity_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_commit = _posix.commit_noreplace
    real_sync = _posix.sync_descriptor
    synced_entries: list[tuple[int, int]] = []
    committed_parent: tuple[int, int] | None = None

    def record_sync(descriptor: int) -> None:
        entry = os.fstat(descriptor)
        key = (entry.st_dev, entry.st_ino)
        if key == committed_parent:
            raise AssertionError("container sync ran after identity commit")
        synced_entries.append(key)
        real_sync(descriptor)

    def record_commit(
        parent_fd: int,
        source: str,
        destination: str,
        expected_source: os.stat_result,
    ) -> _posix._CommitOutcome:
        nonlocal committed_parent
        parent = os.fstat(parent_fd)
        container_key = (parent.st_dev, parent.st_ino)
        container = hidden_entries(tmp_path)[0]
        lock = (container / ".lock").stat(follow_symlinks=False)
        work = (container / "work").stat(follow_symlinks=False)
        parent_key = (tmp_path.stat().st_dev, tmp_path.stat().st_ino)
        lock_key = (lock.st_dev, lock.st_ino)
        work_key = (work.st_dev, work.st_ino)
        assert lock_key in synced_entries
        assert work_key in synced_entries
        assert container_key in synced_entries
        assert parent_key in synced_entries
        internal_sync = max(synced_entries.index(lock_key), synced_entries.index(work_key))
        hierarchy_sync = synced_entries.index(container_key, internal_sync + 1)
        assert hierarchy_sync < synced_entries.index(parent_key)
        outcome = real_commit(parent_fd, source, destination, expected_source)
        committed_parent = container_key
        return outcome

    monkeypatch.setattr(_posix, "sync_descriptor", record_sync)
    monkeypatch.setattr(_posix, "commit_noreplace", record_commit)

    with Workspace(tmp_path / "result", identity=b"request"):
        pass

    assert committed_parent is not None


def test_workspace_reopen_does_not_sync_unchanged_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "result"
    with Workspace(destination, identity=b"request"):
        pass
    container = hidden_entries(tmp_path)[0]
    container_entry = container.stat(follow_symlinks=False)
    container_key = (container_entry.st_dev, container_entry.st_ino)
    synced_entries: list[tuple[int, int]] = []
    real_sync = _posix.sync_descriptor

    def record_sync(descriptor: int) -> None:
        entry = os.fstat(descriptor)
        synced_entries.append((entry.st_dev, entry.st_ino))
        real_sync(descriptor)

    monkeypatch.setattr(_posix, "sync_descriptor", record_sync)

    with Workspace(destination, identity=b"request"):
        pass

    assert container_key not in synced_entries


def test_identity_initialization_failure_allows_same_identity_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "result"
    real_sync = _posix.sync_descriptor
    failed = False

    def fail_first_sync(descriptor: int) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected identity sync failure")
        real_sync(descriptor)

    monkeypatch.setattr(_posix, "sync_descriptor", fail_first_sync)
    with (
        pytest.raises(OSError, match="identity sync failure"),
        Workspace(destination, identity=b"request"),
    ):
        pass

    with Workspace(destination, identity=b"request") as workspace:
        (workspace.path / "checkpoint").write_text("resumed")

    container = hidden_entries(tmp_path)[0]
    assert (container / ".identity").is_file()
    assert list(container.glob(".identity-*.tmp")) == []


def test_corrupt_identity_is_rejected_without_unbounded_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "result"
    with Workspace(destination, identity=b"request"):
        pass
    identity = hidden_entries(tmp_path)[0] / ".identity"
    identity.write_bytes(b"x" * 1_000_000)

    def unexpected_read(descriptor: int, size: int) -> bytes:
        del descriptor, size
        raise AssertionError("oversized identity must be rejected before reading")

    monkeypatch.setattr(_workspace.os, "read", unexpected_read)
    with pytest.raises(WorkConflict), Workspace(destination, identity=b"request"):
        pass


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
