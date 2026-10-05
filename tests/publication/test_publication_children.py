from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
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


def _hold_child(destination: str, name: str, ready: Connection, release: Connection) -> None:
    try:
        with Workspace(destination, identity=b"study").child(name, identity=name.encode()):
            ready.send("entered")
            release.recv()
    except BaseException as error:
        ready.send(type(error).__name__)
        raise


def _spawn_holder(destination: Path, name: str) -> tuple[BaseProcess, Connection, Connection]:
    context = multiprocessing.get_context("spawn")
    ready_parent, ready_child = context.Pipe(duplex=False)
    release_child, release_parent = context.Pipe(duplex=False)
    process = context.Process(
        target=_hold_child, args=(str(destination), name, ready_child, release_child)
    )
    process.start()
    ready_child.close()
    release_child.close()
    return process, ready_parent, release_parent


def _finish(process: BaseProcess, release: Connection) -> None:
    release.send("release")
    release.close()
    process.join(timeout=10)
    assert process.exitcode == 0


def _race_parent(destination: str, start: Connection, result: Connection) -> None:
    start.recv()
    try:
        with Workspace(destination, identity=b"study") as workspace:
            workspace.publish(lambda draft: None)
    except Busy:
        result.send("busy")
    else:
        result.send("published")


def _race_child(destination: str, start: Connection, result: Connection) -> None:
    start.recv()
    try:
        with Workspace(destination, identity=b"study").child("method-0", identity=b"method-0"):
            pass
    except Busy:
        result.send("busy")
    except DestinationExists:
        result.send("finalized")
    else:
        result.send("entered")


def _race_child_publication(destination: str, start: Connection, result: Connection) -> None:
    start.recv()
    try:
        workspace = Workspace(destination, identity=b"study").child(
            "method-0", identity=b"method-0"
        )
        with workspace as child:
            (child.path / "result").write_text("complete")
            child.publish(lambda draft: draft.link(child.path / "result", "result"))
    except Busy:
        result.send("busy")
    else:
        result.send("published")


Racer = Callable[[str, Connection, Connection], None]


def _start(target: Racer, destination: Path) -> tuple[BaseProcess, Connection, Connection]:
    context = multiprocessing.get_context("spawn")
    start_child, start_parent = context.Pipe(duplex=False)
    result_parent, result_child = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(str(destination), start_child, result_child))
    process.start()
    start_child.close()
    result_child.close()
    return process, start_parent, result_parent


def _race(first: Racer, second: Racer, destination: Path) -> tuple[str, str]:
    racers = [_start(first, destination), _start(second, destination)]
    for _, start, _ in racers:
        start.send("start")
        start.close()
    outcomes: list[str] = []
    for process, _, result in racers:
        assert result.poll(10)
        outcomes.append(result.recv())
        process.join(timeout=10)
        assert process.exitcode == 0
    return outcomes[0], outcomes[1]


@pytest.mark.parametrize("name", ["", ".", "..", "nested/child", "child\0suffix"])
def test_child_rejects_unsafe_leaf(tmp_path: Path, name: str) -> None:
    parent = Workspace(tmp_path / "study", identity=b"study")

    with pytest.raises(ConfigurationError, match="unsafe filesystem leaf"):
        parent.child(name, identity=b"candidate")

    assert list(tmp_path.iterdir()) == []


def test_child_rejects_recursive_hierarchy(tmp_path: Path) -> None:
    child = Workspace(tmp_path / "study", identity=b"study").child("method-0", identity=b"m")

    with pytest.raises(ConfigurationError, match="cannot contain child"):
        child.child("nested", identity=b"nested")

    assert list(tmp_path.iterdir()) == []


def test_child_paths_live_under_parent_work(tmp_path: Path) -> None:
    parent = Workspace(tmp_path / "study", identity=b"study")
    child = parent.child("method-0", identity=b"m")

    assert child.path.parent.parent == parent.path
    assert child.path.name == "work"


def test_spawned_sibling_children_overlap_and_exclude_parent(tmp_path: Path) -> None:
    destination = tmp_path / "study"
    first, first_ready, first_release = _spawn_holder(destination, "method-0")
    second, second_ready, second_release = _spawn_holder(destination, "method-1")
    try:
        assert first_ready.poll(10)
        assert first_ready.recv() == "entered"
        assert second_ready.poll(10)
        assert second_ready.recv() == "entered"
        with pytest.raises(Busy, match="already in use"), Workspace(destination, identity=b"study"):
            pass
    finally:
        _finish(first, first_release)
        _finish(second, second_release)


def test_spawned_child_excludes_duplicate_and_parent_then_parent_reopens(tmp_path: Path) -> None:
    destination = tmp_path / "study"
    process, ready, release = _spawn_holder(destination, "method-0")
    try:
        assert ready.poll(10)
        assert ready.recv() == "entered"
        duplicate = Workspace(destination, identity=b"study").child(
            "method-0", identity=b"method-0"
        )
        with pytest.raises(Busy, match="already in use"), duplicate:
            pass
        with pytest.raises(Busy, match="already in use"), Workspace(destination, identity=b"study"):
            pass
    finally:
        _finish(process, release)

    with Workspace(destination, identity=b"study"):
        pass


def test_parent_entry_excludes_child_without_waiting(tmp_path: Path) -> None:
    destination = tmp_path / "study"
    with (
        Workspace(destination, identity=b"study"),
        pytest.raises(Busy, match="already in use"),
        Workspace(destination, identity=b"study").child("method-0", identity=b"method-0"),
    ):
        pass


def test_child_verifies_parent_and_child_identities(tmp_path: Path) -> None:
    destination = tmp_path / "study"
    with Workspace(destination, identity=b"study"):
        pass

    with (
        pytest.raises(WorkspaceConflict, match="different identity"),
        Workspace(destination, identity=b"other").child("method-0", identity=b"method-0"),
    ):
        pass
    with Workspace(destination, identity=b"study").child("method-0", identity=b"method-0"):
        pass
    with (
        pytest.raises(WorkspaceConflict, match="different identity"),
        Workspace(destination, identity=b"study").child("method-0", identity=b"other"),
    ):
        pass


def test_failed_child_resumes_and_publishes_for_parent_assembly(tmp_path: Path) -> None:
    destination = tmp_path / "study"
    parent = Workspace(destination, identity=b"study")
    with parent.child("method-0", identity=b"method-0") as child:
        checkpoint = child.path / "last.ckpt"
        checkpoint.write_bytes(b"checkpoint")

        def fail(draft: Draft) -> None:
            draft.link(checkpoint, "result.bin")
            raise RuntimeError("interrupted")

        with pytest.raises(RuntimeError, match="interrupted"):
            child.publish(fail)

    with parent.child("method-0", identity=b"method-0") as child:
        checkpoint = child.path / "last.ckpt"
        assert checkpoint.read_bytes() == b"checkpoint"
        publication = child.publish(lambda draft: draft.link(checkpoint, "result.bin"))

    completed = parent.path / "method-0"
    assert publication.destination == completed
    assert (completed / "result.bin").read_bytes() == b"checkpoint"
    with parent as workspace:
        workspace.publish(lambda draft: draft.link(completed / "result.bin", "method-0/result.bin"))

    assert (destination / "method-0/result.bin").read_bytes() == b"checkpoint"
    assert list(tmp_path.glob(".servatus-*")) == []


def test_parent_failure_preserves_completed_children(tmp_path: Path) -> None:
    destination = tmp_path / "study"
    parent = Workspace(destination, identity=b"study")
    with parent.child("method-0", identity=b"method-0") as child:
        (child.path / "result.bin").write_bytes(b"complete")
        child.publish(lambda draft: draft.link(child.path / "result.bin", "result.bin"))

    def fail(draft: Draft) -> None:
        draft.link(parent.path / "method-0/result.bin", "method-0/result.bin")
        raise ValueError("invalid study")

    with parent as workspace, pytest.raises(ValueError, match="invalid study"):
        workspace.publish(fail)

    assert (parent.path / "method-0/result.bin").read_bytes() == b"complete"
    assert not destination.exists()


def test_parent_destination_collision_preserves_completed_children(tmp_path: Path) -> None:
    destination = tmp_path / "study"
    parent = Workspace(destination, identity=b"study")
    with parent.child("method-0", identity=b"method-0") as child:
        (child.path / "result").write_text("complete")
        child.publish(lambda draft: draft.link(child.path / "result", "result"))

    with parent as workspace:
        destination.mkdir()
        with pytest.raises(DestinationExists, match="already exists"):
            workspace.publish(
                lambda draft: draft.link(parent.path / "method-0/result", "method-0/result")
            )

    assert (parent.path / "method-0/result").read_text() == "complete"


def test_child_destination_collision_preserves_resumable_work(tmp_path: Path) -> None:
    parent = Workspace(tmp_path / "study", identity=b"study")
    with parent.child("method-0", identity=b"method-0") as child:
        checkpoint = child.path / "checkpoint"
        checkpoint.write_text("resumable")
        (parent.path / "method-0").mkdir()
        with pytest.raises(DestinationExists, match="already exists"):
            child.publish(lambda draft: draft.link(checkpoint, "result"))

    assert checkpoint.read_text() == "resumable"


def test_final_destination_blocks_child_without_reclaiming_parent(tmp_path: Path) -> None:
    destination = tmp_path / "study"
    parent = Workspace(destination, identity=b"study")
    child = parent.child("method-0", identity=b"method-0")
    with child:
        (child.path / "checkpoint").write_text("stale")
    destination.mkdir()

    with (
        pytest.raises(DestinationExists, match="already exists"),
        parent.child("method-1", identity=b"method-1"),
    ):
        pass

    assert (child.path / "checkpoint").read_text() == "stale"


def test_completed_child_destination_reclaims_stale_duplicate(tmp_path: Path) -> None:
    parent = Workspace(tmp_path / "study", identity=b"study")
    with parent.child("method-0", identity=b"method-0") as child:
        (child.path / "source").write_text("complete")
        child.publish(lambda draft: draft.link(child.path / "source", "result"))

    with (
        pytest.raises(DestinationExists, match="already exists"),
        parent.child("method-0", identity=b"method-0"),
    ):
        pass

    assert sorted(path.name for path in parent.path.iterdir()) == ["method-0"]


def test_child_discard_removes_only_that_child(tmp_path: Path) -> None:
    parent = Workspace(tmp_path / "study", identity=b"study")
    keep = parent.child("keep", identity=b"keep")
    with keep:
        (keep.path / "value").write_text("kept")
    with parent.child("drop", identity=b"drop") as drop:
        (drop.path / "value").write_text("dropped")
        drop.discard()

    with keep:
        assert (keep.path / "value").read_text() == "kept"
    assert len(list(parent.path.iterdir())) == 1


def test_active_child_rejects_container_replacement(tmp_path: Path) -> None:
    parent = Workspace(tmp_path / "study", identity=b"study")
    with parent.child("method-0", identity=b"method-0") as child:
        container = child.path.parent
        container.rename(container.with_name(f"{container.name}-moved"))
        container.mkdir(mode=0o700)
        with pytest.raises(UnsafeFilesystem, match="substituted"):
            child.publish(lambda draft: None)

    assert not (tmp_path / "study").exists()


def test_replaced_child_lock_cannot_admit_duplicate(tmp_path: Path) -> None:
    parent = Workspace(tmp_path / "study", identity=b"study")
    with parent.child("method-0", identity=b"method-0") as child:
        lock = child.path.parent / ".lock"
        lock.rename(lock.with_name(".lock-moved"))
        lock.write_text("replacement")
        lock.chmod(0o600)
        with (
            pytest.raises(UnsafeFilesystem, match="lifecycle entries changed"),
            parent.child("method-0", identity=b"method-0"),
        ):
            pass


def test_replaced_parent_lock_cannot_admit_parent_during_child(tmp_path: Path) -> None:
    destination = tmp_path / "study"
    parent = Workspace(destination, identity=b"study")
    with parent.child("method-0", identity=b"method-0"):
        lock = parent.path.parent / ".lock"
        lock.rename(lock.with_name(".lock-moved"))
        lock.write_text("replacement")
        lock.chmod(0o600)
        with (
            pytest.raises(UnsafeFilesystem, match="lifecycle entries changed"),
            Workspace(destination, identity=b"study"),
        ):
            pass


def test_spawned_parent_finalization_and_child_open_race_is_fail_closed(tmp_path: Path) -> None:
    for index in range(8):
        destination = tmp_path / f"study-{index}"
        with Workspace(destination, identity=b"study"):
            pass

        parent_outcome, child_outcome = _race(_race_parent, _race_child, destination)

        assert parent_outcome in {"busy", "published"}
        assert child_outcome in {"busy", "entered", "finalized"}
        assert parent_outcome == "published" or child_outcome == "entered"
        assert destination.exists() is (parent_outcome == "published")


def test_spawned_child_cleanup_and_duplicate_open_race_is_fail_closed(tmp_path: Path) -> None:
    for index in range(8):
        destination = tmp_path / f"study-{index}"
        parent = Workspace(destination, identity=b"study")
        with parent.child("method-0", identity=b"method-0"):
            pass

        publisher, opener = _race(_race_child_publication, _race_child, destination)

        assert publisher in {"busy", "published"}
        assert opener in {"busy", "entered", "finalized"}
        assert (parent.path / "method-0").exists() is (publisher == "published")
