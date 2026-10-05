from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from servatus import _fs
from servatus.errors import Unavailable, UnsafeFilesystem
from servatus.publication import Workspace, publish, publish_file


def open_descriptors() -> int:
    return len(os.listdir("/dev/fd"))


def hidden(parent: Path) -> list[Path]:
    return sorted(parent.glob(".servatus-*"))


def track_opens(syscalls: Any, prefix: str) -> set[int]:
    """Record descriptors opened for entries whose name starts with `prefix`."""
    opened: set[int] = set()

    def open_(real: Callable[..., int], path: Any, *args: Any, **kwargs: Any) -> int:
        fd = real(path, *args, **kwargs)
        if str(path).startswith(prefix):
            opened.add(fd)
        return fd

    syscalls.os("open", open_)
    return opened


@pytest.mark.parametrize(
    ("failure", "raised"),
    [(OSError("injected fstat failure"), Unavailable), (KeyboardInterrupt(), KeyboardInterrupt)],
)
def test_fstat_failure_on_a_new_stage_leaks_nothing(
    tmp_path: Path, syscalls: Any, failure: BaseException, raised: type[BaseException]
) -> None:
    # Regression: the descriptor (and stage) leaked when fstat failed right after open.
    opened = track_opens(syscalls, ".servatus-stage-")

    def fstat(real: Callable[[int], os.stat_result], fd: int) -> os.stat_result:
        if fd in opened:
            raise failure
        return real(fd)

    syscalls.os("fstat", fstat)
    before = open_descriptors()

    for publisher in (
        lambda: publish(tmp_path / "directory", lambda draft: None),
        lambda: publish_file(tmp_path / "file", lambda path: path.write_text("x")),
    ):
        with pytest.raises(raised, match="injected fstat failure|^$"):
            publisher()

    assert open_descriptors() == before
    assert list(tmp_path.iterdir()) == []


def test_stat_failure_after_stage_creation_leaves_no_residue(tmp_path: Path, syscalls: Any) -> None:
    def stat(real: Callable[..., os.stat_result], path: Any, *args: Any, **kwargs: Any) -> Any:
        if str(path).startswith(".servatus-stage-"):
            raise KeyboardInterrupt
        return real(path, *args, **kwargs)

    syscalls.os("stat", stat)
    before = open_descriptors()

    with pytest.raises(KeyboardInterrupt):
        publish(tmp_path / "result", lambda draft: None)

    assert open_descriptors() == before
    assert list(tmp_path.iterdir()) == []


def test_fstat_failure_on_workspace_lock_leaks_nothing(tmp_path: Path, syscalls: Any) -> None:
    opened = track_opens(syscalls, ".lock")

    def fstat(real: Callable[[int], os.stat_result], fd: int) -> os.stat_result:
        if fd in opened:
            raise OSError("injected fstat failure")
        return real(fd)

    syscalls.os("fstat", fstat)
    before = open_descriptors()

    with (
        pytest.raises(Unavailable, match="injected fstat failure"),
        Workspace(tmp_path / "result", identity=b"request"),
    ):
        pass

    assert open_descriptors() == before


def test_write_new_failure_removes_the_file_and_descriptor(tmp_path: Path, syscalls: Any) -> None:
    def write(real: Callable[..., int], fd: int, data: Any) -> int:
        raise OSError("injected write failure")

    syscalls.os("write", write)
    before = open_descriptors()

    with _fs.open_path(tmp_path) as directory, pytest.raises(OSError, match="write failure"):
        _fs.write_new(directory.fd, "state", b"payload")

    assert open_descriptors() == before
    assert list(tmp_path.iterdir()) == []


def test_failed_attempt_closes_stage_descriptor_once(tmp_path: Path, syscalls: Any) -> None:
    opened = track_opens(syscalls, ".servatus-stage-")
    closes: list[int] = []

    def close(real: Callable[[int], None], fd: int) -> None:
        if fd in opened:
            closes.append(fd)
        real(fd)

    syscalls.os("close", close)

    with pytest.raises(ValueError, match="stop"):
        publish(tmp_path / "result", lambda draft: (_ for _ in ()).throw(ValueError("stop")))

    assert len(closes) == len(opened) == 1


def test_interrupted_sync_removes_stage_and_leaks_nothing(tmp_path: Path, syscalls: Any) -> None:
    def interrupt(real: Callable[[int], None], fd: int) -> None:
        raise KeyboardInterrupt

    syscalls.on_sync(interrupt)
    before = open_descriptors()

    def build(draft: Any) -> None:
        (draft.path / "nested").mkdir()
        (draft.path / "nested/value").write_text("x")

    with pytest.raises(KeyboardInterrupt):
        publish(tmp_path / "result", build)

    assert open_descriptors() == before
    assert list(tmp_path.iterdir()) == []


def test_many_operations_leak_no_descriptors(tmp_path: Path) -> None:
    before = open_descriptors()
    source = tmp_path / "source"
    source.write_text("value")
    (tmp_path / "tree").mkdir()
    (tmp_path / "tree/value").write_text("value")

    for index in range(5):
        publish(tmp_path / f"d{index}", lambda draft: draft.link_tree(tmp_path / "tree", "copy"))
        publish_file(tmp_path / f"f{index}", lambda path: path.write_text("x"))
        with pytest.raises(ValueError, match="stop"):
            publish(tmp_path / "x", lambda draft: (_ for _ in ()).throw(ValueError("stop")))
        root = Workspace(tmp_path / f"w{index}", identity=b"root")
        with root.child("child", identity=b"child") as child:
            child.publish(lambda draft: draft.link(source, "value"))
        with root as workspace:
            workspace.publish(lambda draft: None)

    assert open_descriptors() == before
    assert hidden(tmp_path) == []


# -- substitution between creation and open --------------------------------------------------
def substitute_on_open(
    syscalls: Any, matches: Callable[[str], bool], moved: Path, parent: Path, fill: bool
) -> list[Path]:
    replaced: list[Path] = []

    def open_(real: Callable[..., int], path: Any, *args: Any, **kwargs: Any) -> int:
        if not replaced and isinstance(path, str) and matches(path):
            original = parent / path
            original.rename(moved)
            original.mkdir(mode=0o700)
            if fill:
                (original / "injected").write_text("preserve")
            replaced.append(original)
        return real(path, *args, **kwargs)

    syscalls.os("open", open_)
    return replaced


@pytest.mark.parametrize("fill", [False, True])
def test_new_stage_substituted_before_open_is_preserved(
    tmp_path: Path, syscalls: Any, fill: bool
) -> None:
    moved = tmp_path / "moved-stage"
    replaced = substitute_on_open(
        syscalls, lambda name: name.startswith(".servatus-stage-"), moved, tmp_path, fill
    )

    with pytest.raises(UnsafeFilesystem, match="substituted before it was opened"):
        publish(tmp_path / "result", lambda draft: None)

    assert moved.is_dir()
    assert replaced[0].is_dir()
    assert not (tmp_path / "result").exists()


def test_new_container_substituted_before_open_is_preserved(tmp_path: Path, syscalls: Any) -> None:
    moved = tmp_path / "moved-container"
    replaced = substitute_on_open(
        syscalls, lambda name: name.endswith(".work"), moved, tmp_path, False
    )

    with (
        pytest.raises(UnsafeFilesystem, match="substituted before it was opened"),
        Workspace(tmp_path / "result", identity=b"request"),
    ):
        pass

    assert moved.is_dir()
    assert replaced[0].is_dir()


@pytest.mark.parametrize("entry", ["container", "work", ".lock", ".identity"])
def test_foreign_owned_private_entries_are_rejected(
    tmp_path: Path, syscalls: Any, entry: str
) -> None:
    with Workspace(tmp_path / "result", identity=b"request") as workspace:
        container = workspace.path.parent
    target = (container if entry == "container" else container / entry).stat().st_ino

    def fstat(real: Callable[[int], os.stat_result], fd: int) -> os.stat_result:
        result = real(fd)
        if result.st_ino != target:
            return result
        fields = list(result)
        fields[4] = result.st_uid + 1
        return os.stat_result(fields)

    syscalls.os("fstat", fstat)

    with (
        pytest.raises(UnsafeFilesystem, match="owner-only"),
        Workspace(tmp_path / "result", identity=b"request"),
    ):
        pass


@pytest.mark.parametrize("failure", [OSError(5, "close reported EIO"), KeyboardInterrupt()])
def test_write_new_close_failure_removes_the_file(
    tmp_path: Path, syscalls: Any, failure: BaseException
) -> None:
    # Regression: the descriptor was closed after the cleanup handler, so the file survived.
    opened = track_opens(syscalls, "state")

    def close(real: Callable[[int], None], fd: int) -> None:
        real(fd)
        if fd in opened:
            raise failure

    syscalls.os("close", close)
    before = open_descriptors()

    with _fs.open_path(tmp_path) as directory, pytest.raises(type(failure), match="EIO|^$"):
        _fs.write_new(directory.fd, "state", b"payload")

    assert open_descriptors() == before
    assert list(tmp_path.iterdir()) == []


def test_stage_name_collision_allocates_another_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    taken = tmp_path / f".servatus-stage-{bytes(12).hex()}"
    taken.mkdir()
    (taken / "foreign").write_text("preserve")
    draws: list[int] = []
    real = os.urandom

    def urandom(size: int) -> bytes:
        draws.append(size)
        return bytes(size) if len(draws) == 1 else real(size)

    monkeypatch.setattr(os, "urandom", urandom)

    publication = publish(tmp_path / "result", lambda draft: (draft.path / "value").touch())

    assert sorted(path.name for path in publication.destination.iterdir()) == ["value"]
    assert sorted(path.name for path in taken.iterdir()) == ["foreign"]
