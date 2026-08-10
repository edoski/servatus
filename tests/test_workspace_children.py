from __future__ import annotations

import multiprocessing
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

from servatus import (
    DestinationExists,
    Draft,
    UnsafePublication,
    WorkConflict,
    Workspace,
    WorkspaceBusy,
)


def _hold_child(
    destination: str,
    name: str,
    ready: Connection,
    release: Connection,
) -> None:
    try:
        with Workspace(Path(destination), identity=b"study").child(name, identity=name.encode()):
            ready.send("entered")
            release.recv()
    except BaseException as error:
        ready.send(type(error).__name__)
        raise


def _spawn_child_holder(
    destination: Path, name: str
) -> tuple[multiprocessing.Process, Connection, Connection]:
    context = multiprocessing.get_context("spawn")
    ready_parent, ready_child = context.Pipe(duplex=False)
    release_child, release_parent = context.Pipe(duplex=False)
    process = context.Process(
        target=_hold_child,
        args=(str(destination), name, ready_child, release_child),
    )
    process.start()
    ready_child.close()
    release_child.close()
    return process, ready_parent, release_parent


def _finish_holder(process: multiprocessing.Process, release: Connection) -> None:
    release.send("release")
    release.close()
    process.join(timeout=10)
    assert process.exitcode == 0


def _race_parent(destination: str, start: Connection, result: Connection) -> None:
    start.recv()
    try:
        with Workspace(Path(destination), identity=b"study") as workspace:
            workspace.publish(lambda draft: None)
    except WorkspaceBusy:
        result.send("busy")
    else:
        result.send("published")


def _race_child(destination: str, start: Connection, result: Connection) -> None:
    start.recv()
    try:
        with Workspace(Path(destination), identity=b"study").child(
            "method-0", identity=b"method-0"
        ):
            pass
    except WorkspaceBusy:
        result.send("busy")
    except DestinationExists:
        result.send("finalized")
    else:
        result.send("entered")


def _race_child_publication(destination: str, start: Connection, result: Connection) -> None:
    start.recv()
    try:
        with Workspace(Path(destination), identity=b"study").child(
            "method-0", identity=b"method-0"
        ) as child:
            source = child.path / "result"
            source.write_text("complete")
            child.publish(lambda draft: draft.link(source, "result"))
    except WorkspaceBusy:
        result.send("busy")
    else:
        result.send("published")


def _start_racer(
    target: object, destination: Path
) -> tuple[multiprocessing.Process, Connection, Connection]:
    context = multiprocessing.get_context("spawn")
    start_child, start_parent = context.Pipe(duplex=False)
    result_parent, result_child = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(str(destination), start_child, result_child))
    process.start()
    start_child.close()
    result_child.close()
    return process, start_parent, result_parent


@pytest.mark.parametrize("name", ["", ".", "..", "nested/child", "child\0suffix"])
def test_child_rejects_unsafe_leaf(tmp_path: Path, name: str) -> None:
    parent = Workspace(tmp_path / "study", identity=b"study")

    with pytest.raises(UnsafePublication):
        parent.child(name, identity=b"candidate")

    assert list(tmp_path.iterdir()) == []


def test_child_rejects_recursive_hierarchy(tmp_path: Path) -> None:
    child = Workspace(tmp_path / "study", identity=b"study").child("method-0", identity=b"method-0")

    with pytest.raises(UnsafePublication):
        child.child("nested", identity=b"nested")

    assert list(tmp_path.iterdir()) == []


def test_spawned_sibling_children_overlap_and_exclude_parent(tmp_path: Path) -> None:
    destination = tmp_path / "study"
    first, first_ready, first_release = _spawn_child_holder(destination, "method-0")
    second, second_ready, second_release = _spawn_child_holder(destination, "method-1")
    try:
        assert first_ready.poll(10)
        assert first_ready.recv() == "entered"
        assert second_ready.poll(10)
        assert second_ready.recv() == "entered"
        with pytest.raises(WorkspaceBusy), Workspace(destination, identity=b"study"):
            pass
    finally:
        _finish_holder(first, first_release)
        _finish_holder(second, second_release)


def test_spawned_child_excludes_duplicate_and_parent_then_parent_reopens(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "study"
    process, ready, release = _spawn_child_holder(destination, "method-0")
    try:
        assert ready.poll(10)
        assert ready.recv() == "entered"
        with (
            pytest.raises(WorkspaceBusy),
            Workspace(destination, identity=b"study").child("method-0", identity=b"method-0"),
        ):
            pass
        with pytest.raises(WorkspaceBusy), Workspace(destination, identity=b"study"):
            pass
    finally:
        _finish_holder(process, release)

    with Workspace(destination, identity=b"study"):
        pass


def test_parent_entry_excludes_child_without_waiting(tmp_path: Path) -> None:
    destination = tmp_path / "study"

    with (
        Workspace(destination, identity=b"study"),
        pytest.raises(WorkspaceBusy),
        Workspace(destination, identity=b"study").child("method-0", identity=b"method-0"),
    ):
        pass


def test_child_verifies_parent_and_child_identities(tmp_path: Path) -> None:
    destination = tmp_path / "study"
    with Workspace(destination, identity=b"study"):
        pass

    with (
        pytest.raises(WorkConflict),
        Workspace(destination, identity=b"other-study").child("method-0", identity=b"method-0"),
    ):
        pass

    with Workspace(destination, identity=b"study").child("method-0", identity=b"method-0"):
        pass
    with (
        pytest.raises(WorkConflict),
        Workspace(destination, identity=b"study").child("method-0", identity=b"other-method"),
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
        child.publish(lambda draft: draft.link(checkpoint, "result.bin"))

    completed_child = parent.path / "method-0"
    assert (completed_child / "result.bin").read_bytes() == b"checkpoint"
    with parent as workspace:
        workspace.publish(
            lambda draft: draft.link(completed_child / "result.bin", "method-0/result.bin")
        )

    assert (destination / "method-0/result.bin").read_bytes() == b"checkpoint"
    assert list(tmp_path.glob(".servatus-*")) == []


def test_parent_failure_preserves_completed_children(tmp_path: Path) -> None:
    destination = tmp_path / "study"
    parent = Workspace(destination, identity=b"study")
    with parent.child("method-0", identity=b"method-0") as child:
        source = child.path / "result.bin"
        source.write_bytes(b"complete")
        child.publish(lambda draft: draft.link(source, "result.bin"))

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
        source = child.path / "result"
        source.write_text("complete")
        child.publish(lambda draft: draft.link(source, "result"))

    with parent as workspace:
        destination.mkdir()
        with pytest.raises(DestinationExists):
            workspace.publish(
                lambda draft: draft.link(parent.path / "method-0/result", "method-0/result")
            )

    assert (parent.path / "method-0/result").read_text() == "complete"
    assert destination.is_dir()


def test_child_destination_collision_preserves_resumable_work(tmp_path: Path) -> None:
    parent = Workspace(tmp_path / "study", identity=b"study")
    child = parent.child("method-0", identity=b"method-0")
    with child as workspace:
        checkpoint = workspace.path / "checkpoint"
        checkpoint.write_text("resumable")
        (parent.path / "method-0").mkdir()
        with pytest.raises(DestinationExists):
            workspace.publish(lambda draft: draft.link(checkpoint, "result"))

    assert checkpoint.read_text() == "resumable"


def test_final_destination_blocks_stale_child_state(tmp_path: Path) -> None:
    destination = tmp_path / "study"
    parent = Workspace(destination, identity=b"study")
    child = parent.child("method-0", identity=b"method-0")
    stale_checkpoint = child.path / "checkpoint"
    with child:
        stale_checkpoint.write_text("stale")
    destination.mkdir()

    with (
        pytest.raises(DestinationExists),
        parent.child("method-1", identity=b"method-1"),
    ):
        pass

    assert destination.is_dir()
    assert stale_checkpoint.read_text() == "stale"


def test_completed_child_destination_blocks_stale_duplicate(tmp_path: Path) -> None:
    parent = Workspace(tmp_path / "study", identity=b"study")
    with parent.child("method-0", identity=b"method-0") as child:
        source = child.path / "source"
        source.write_text("complete")
        child.publish(lambda draft: draft.link(source, "result"))

    with (
        pytest.raises(DestinationExists),
        parent.child("method-0", identity=b"method-0"),
    ):
        pass


@pytest.mark.parametrize("substitution", ["container", "lock", "work"])
def test_child_rejects_lifecycle_entry_substitution(tmp_path: Path, substitution: str) -> None:
    parent = Workspace(tmp_path / "study", identity=b"study")
    with parent.child("method-0", identity=b"method-0") as child:
        container = child.path.parent
        if substitution == "container":
            container.rename(container.with_name(f"{container.name}-moved"))
            container.mkdir()
        else:
            entry = container / (".lock" if substitution == "lock" else "work")
            entry.rename(container / f"{entry.name}-moved")
            if substitution == "lock":
                entry.write_text("replacement")
            else:
                entry.mkdir()
        with pytest.raises(UnsafePublication):
            child.publish(lambda draft: None)

    assert not (tmp_path / "study").exists()


def test_spawned_parent_finalization_and_child_open_race_is_fail_closed(
    tmp_path: Path,
) -> None:
    for index in range(8):
        destination = tmp_path / f"study-{index}"
        with Workspace(destination, identity=b"study"):
            pass
        parent_process, parent_start, parent_result = _start_racer(_race_parent, destination)
        child_process, child_start, child_result = _start_racer(_race_child, destination)

        parent_start.send("start")
        child_start.send("start")
        parent_start.close()
        child_start.close()
        assert parent_result.poll(10)
        assert child_result.poll(10)
        parent_outcome = parent_result.recv()
        child_outcome = child_result.recv()
        parent_process.join(timeout=10)
        child_process.join(timeout=10)
        assert parent_process.exitcode == 0
        assert child_process.exitcode == 0

        assert parent_outcome in {"busy", "published"}
        assert child_outcome in {"busy", "entered", "finalized"}
        assert parent_outcome == "published" or child_outcome == "entered"
        assert destination.exists() is (parent_outcome == "published")


def test_spawned_child_cleanup_and_duplicate_open_race_is_fail_closed(
    tmp_path: Path,
) -> None:
    for index in range(8):
        destination = tmp_path / f"study-{index}"
        parent = Workspace(destination, identity=b"study")
        with parent.child("method-0", identity=b"method-0"):
            pass
        publisher, publisher_start, publisher_result = _start_racer(
            _race_child_publication, destination
        )
        opener, opener_start, opener_result = _start_racer(_race_child, destination)

        publisher_start.send("start")
        opener_start.send("start")
        publisher_start.close()
        opener_start.close()
        assert publisher_result.poll(10)
        assert opener_result.poll(10)
        publisher_outcome = publisher_result.recv()
        opener_outcome = opener_result.recv()
        publisher.join(timeout=10)
        opener.join(timeout=10)
        assert publisher.exitcode == 0
        assert opener.exitcode == 0

        assert publisher_outcome in {"busy", "published"}
        assert opener_outcome in {"busy", "entered", "finalized"}
        completed = parent.path / "method-0"
        assert completed.exists() is (publisher_outcome == "published")
