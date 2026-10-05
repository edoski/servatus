from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import pytest

from servatus.errors import (
    Busy,
    ConfigurationError,
    DestinationExists,
    UnsafeFilesystem,
    WorkspaceConflict,
)
from servatus.publication import Draft, Workspace


def hidden(parent: Path) -> list[Path]:
    return sorted(parent.glob(".servatus-*"))


def container_of(tmp_path: Path, destination: str = "result") -> Path:
    with Workspace(tmp_path / destination, identity=b"request") as workspace:
        return workspace.path.parent


def test_workspace_preserves_failure_and_resumes_same_identity(tmp_path: Path) -> None:
    destination = tmp_path / "result"
    with Workspace(destination, identity=b"request") as workspace:
        checkpoint = workspace.path / "last.ckpt"
        checkpoint.write_bytes(b"checkpoint")

        def fail(draft: Draft) -> None:
            draft.link(checkpoint, "last.ckpt")
            raise RuntimeError("not complete")

        with pytest.raises(RuntimeError, match="not complete"):
            workspace.publish(fail)

    with Workspace(str(destination), identity=b"request") as workspace:
        assert (workspace.path / "last.ckpt").read_bytes() == b"checkpoint"
        publication = workspace.publish(
            lambda draft: draft.link(workspace.path / "last.ckpt", "last.ckpt")
        )

    assert publication.destination == destination
    assert publication.cleanup_pending is False
    assert (destination / "last.ckpt").read_bytes() == b"checkpoint"
    assert hidden(tmp_path) == []


def test_interrupted_builder_preserves_resumable_work(tmp_path: Path) -> None:
    destination = tmp_path / "result"

    def interrupt(draft: Draft) -> None:
        (draft.path / "partial").write_bytes(b"x")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt), Workspace(destination, identity=b"run") as workspace:
        (workspace.path / "last.ckpt").write_bytes(b"ckpt")
        workspace.publish(interrupt)

    assert [path.name.startswith(".servatus-") for path in tmp_path.iterdir()] == [True]
    with Workspace(destination, identity=b"run") as workspace:
        assert (workspace.path / "last.ckpt").read_bytes() == b"ckpt"


def test_failed_publication_may_be_retried_in_the_same_session(tmp_path: Path) -> None:
    with Workspace(tmp_path / "result", identity=b"request") as workspace:
        with pytest.raises(ValueError, match="first"):
            workspace.publish(lambda draft: (_ for _ in ()).throw(ValueError("first")))
        publication = workspace.publish(lambda draft: None)

    assert publication.cleanup_pending is False


def test_workspace_publication_applies_mode(tmp_path: Path) -> None:
    with Workspace(tmp_path / "result", identity=b"request") as workspace:
        workspace.publish(lambda draft: (draft.path / "child").mkdir(), mode=0o750)

    assert stat.S_IMODE((tmp_path / "result").stat().st_mode) == 0o750
    assert stat.S_IMODE((tmp_path / "result/child").stat().st_mode) == 0o750


def test_session_rules_are_enforced(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "result", identity=b"request")
    with pytest.raises(RuntimeError, match="entered first"):
        workspace.publish(lambda draft: None)
    with pytest.raises(RuntimeError, match="entered first"):
        workspace.discard()
    with workspace:
        with pytest.raises(RuntimeError, match="already entered"), workspace:
            pass
        workspace.publish(lambda draft: None)
        with pytest.raises(RuntimeError, match="already published"):
            workspace.publish(lambda draft: None)
        with pytest.raises(RuntimeError, match="already published"):
            workspace.discard()


def test_identity_must_be_bytes(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="identity must be bytes"):
        Workspace(tmp_path / "result", identity="text")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="identity must be bytes"):
        Workspace(tmp_path / "result", identity=b"root").child("a", identity=1)  # type: ignore[arg-type]


def test_identity_record_stores_canonical_inode_pins(tmp_path: Path) -> None:
    container = container_of(tmp_path)
    entries = (container, container / ".lock", container / "work")
    expected = (
        b"servatus-workspace-v2\n"
        + hashlib.sha256(b"request").hexdigest().encode("ascii")
        + b"\n"
        + b"".join(f"{entry.stat(follow_symlinks=False).st_ino}\n".encode() for entry in entries)
    )

    assert (container / ".identity").read_bytes() == expected
    for entry in (*entries, container / ".identity"):
        assert stat.S_IMODE(entry.stat().st_mode) & 0o077 == 0


@pytest.mark.parametrize(
    ("entry", "mode"),
    [("container", 0o750), ("work", 0o750), (".lock", 0o640), (".identity", 0o640)],
)
def test_permissive_private_entries_are_rejected(tmp_path: Path, entry: str, mode: int) -> None:
    container = container_of(tmp_path)
    (container if entry == "container" else container / entry).chmod(mode)

    with (
        pytest.raises(UnsafeFilesystem, match="owner-only"),
        Workspace(tmp_path / "result", identity=b"request"),
    ):
        pass


@pytest.mark.parametrize(
    ("entry", "mode"), [("container", 0o770), ("work", 0o770), (".lock", 0o660)]
)
def test_permissions_changed_after_entry_block_publication(
    tmp_path: Path, entry: str, mode: int
) -> None:
    with Workspace(tmp_path / "result", identity=b"request") as workspace:
        container = workspace.path.parent
        (container if entry == "container" else container / entry).chmod(mode)
        with pytest.raises(UnsafeFilesystem, match="owner-only"):
            workspace.publish(lambda draft: None)

    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("pin", [0, 1, 2])
def test_changed_persisted_inode_is_rejected(tmp_path: Path, pin: int) -> None:
    record = container_of(tmp_path) / ".identity"
    header, digest, *pins = record.read_bytes().splitlines()
    pins[pin] = str(int(pins[pin]) + 1).encode("ascii")
    record.write_bytes(b"\n".join((header, digest, *pins)) + b"\n")

    with (
        pytest.raises(UnsafeFilesystem, match="lifecycle entries changed"),
        Workspace(tmp_path / "result", identity=b"request"),
    ):
        pass


@pytest.mark.parametrize(
    "pins",
    [b"2\n4\n", b"inode\n4\n6\n", b"02\n4\n6\n", b"18446744073709551616\n4\n6\n", b"2\n4\n6\nx\n"],
)
def test_malformed_persisted_pins_are_rejected(tmp_path: Path, pins: bytes) -> None:
    record = container_of(tmp_path) / ".identity"
    header, digest, *_ = record.read_bytes().splitlines(keepends=True)
    record.write_bytes(header + digest + pins)

    with (
        pytest.raises(UnsafeFilesystem, match="lifecycle entries changed"),
        Workspace(tmp_path / "result", identity=b"request"),
    ):
        pass


def test_oversized_identity_is_rejected_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (container_of(tmp_path) / ".identity").write_bytes(b"x" * 1_000_000)

    def unexpected_read(fd: int, size: int) -> bytes:
        raise AssertionError("oversized identity must be rejected before reading")

    monkeypatch.setattr(os, "read", unexpected_read)
    with (
        pytest.raises(WorkspaceConflict, match="identity is invalid"),
        Workspace(tmp_path / "result", identity=b"request"),
    ):
        pass


def test_different_identity_conflict_names_the_private_path(tmp_path: Path) -> None:
    with Workspace(tmp_path / "result", identity=b"request-a") as workspace:
        (workspace.path / "checkpoint").write_text("state")
        private = workspace.path

    with (
        pytest.raises(WorkspaceConflict, match="different identity") as raised,
        Workspace(tmp_path / "result", identity=b"request-b"),
    ):
        pass

    assert str(private) in str(raised.value)
    assert (private / "checkpoint").read_text() == "state"


def test_lifecycle_lease_is_nonblocking(tmp_path: Path) -> None:
    with (
        Workspace(tmp_path / "result", identity=b"request"),
        pytest.raises(Busy, match="already in use"),
        Workspace(tmp_path / "result", identity=b"request"),
    ):
        pass


def test_parent_path_substitution_blocks_publication(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    moved = tmp_path / "moved-parent"

    with Workspace(parent / "result", identity=b"request") as workspace:
        parent.rename(moved)
        parent.mkdir()
        with pytest.raises(UnsafeFilesystem, match="substituted"):
            workspace.publish(lambda draft: None)

    assert not (parent / "result").exists()
    assert not (moved / "result").exists()


@pytest.mark.parametrize("entry", [".lock", "work"])
def test_lifecycle_entry_substitution_blocks_publication(tmp_path: Path, entry: str) -> None:
    with Workspace(tmp_path / "result", identity=b"request") as workspace:
        original = workspace.path.parent / entry
        original.rename(original.with_name(f"{entry}-moved"))
        if entry == ".lock":
            original.write_text("replacement")
            original.chmod(0o600)
        else:
            original.mkdir(mode=0o700)
        with pytest.raises(UnsafeFilesystem, match="substituted"):
            workspace.publish(lambda draft: None)

    assert not (tmp_path / "result").exists()


def test_cleanup_removes_unreadable_files_and_read_only_directories(tmp_path: Path) -> None:
    with Workspace(tmp_path / "result", identity=b"request") as workspace:
        (workspace.path / "private").write_text("resumable")
        (workspace.path / "private").chmod(0o000)
        (workspace.path / "frozen").mkdir()
        (workspace.path / "frozen/value").write_text("x")
        (workspace.path / "frozen").chmod(0o500)
        publication = workspace.publish(lambda draft: None)

    assert publication.cleanup_pending is False
    assert hidden(tmp_path) == []


def test_nested_independent_workspace_can_feed_parent_publication(tmp_path: Path) -> None:
    destination = tmp_path / "study"
    with Workspace(destination, identity=b"study") as parent:
        trial_destination = parent.path / "trial-0"
        with Workspace(trial_destination, identity=b"trial") as trial:
            (trial.path / "result.bin").write_bytes(b"trial result")
            trial.publish(lambda draft: draft.link(trial.path / "result.bin", "result.bin"))
        parent.publish(
            lambda draft: draft.link(trial_destination / "result.bin", "trial-0/result.bin")
        )

    assert (destination / "trial-0/result.bin").read_bytes() == b"trial result"
    assert hidden(tmp_path) == []


# -- discard and reclaim ---------------------------------------------------------------------
def test_discard_removes_private_work_and_allows_a_fresh_start(tmp_path: Path) -> None:
    destination = tmp_path / "result"
    with Workspace(destination, identity=b"request") as workspace:
        (workspace.path / "checkpoint").write_text("stale")
        (workspace.path / "frozen").mkdir(mode=0o500)
        workspace.discard()
        assert hidden(tmp_path) == []
        with pytest.raises(RuntimeError, match="already discarded"):
            workspace.publish(lambda draft: None)

    with Workspace(destination, identity=b"other") as workspace:
        assert list(workspace.path.iterdir()) == []
    assert not destination.exists()


def test_existing_destination_reclaims_same_identity_work(tmp_path: Path) -> None:
    destination = tmp_path / "result"
    with Workspace(destination, identity=b"request") as workspace:
        (workspace.path / "checkpoint").write_text("redundant")
    destination.mkdir()

    with (
        pytest.raises(DestinationExists, match="already exists"),
        Workspace(destination, identity=b"request"),
    ):
        pytest.fail("redundant work was exposed")

    assert hidden(tmp_path) == []


def test_existing_destination_keeps_other_identity_work(tmp_path: Path) -> None:
    destination = tmp_path / "result"
    with Workspace(destination, identity=b"request-a") as workspace:
        (workspace.path / "checkpoint").write_text("other work")
        private = workspace.path
    destination.mkdir()

    with (
        pytest.raises(DestinationExists, match="already exists") as raised,
        Workspace(destination, identity=b"request-b"),
    ):
        pass

    assert (private / "checkpoint").read_text() == "other work"
    assert any(str(private) in note for note in raised.value.__notes__)


def test_existing_destination_does_not_reclaim_busy_work(tmp_path: Path) -> None:
    destination = tmp_path / "result"
    with Workspace(destination, identity=b"request") as holder:
        (holder.path / "checkpoint").write_text("in use")
        destination.mkdir()
        with (
            pytest.raises(DestinationExists, match="already exists"),
            Workspace(destination, identity=b"request"),
        ):
            pass
        assert (holder.path / "checkpoint").read_text() == "in use"


def test_existing_destination_without_private_work_creates_nothing(tmp_path: Path) -> None:
    (tmp_path / "result").mkdir()

    with (
        pytest.raises(DestinationExists, match="already exists"),
        Workspace(tmp_path / "result", identity=b"request"),
    ):
        pass

    assert sorted(path.name for path in tmp_path.iterdir()) == ["result"]
