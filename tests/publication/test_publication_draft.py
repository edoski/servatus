from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from servatus.errors import ConfigurationError, DestinationExists, UnsafeFilesystem
from servatus.publication import Draft, publish


def test_link_creates_nested_regular_file_with_same_inode(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"value")

    publish(tmp_path / "result", lambda draft: draft.link(source, "nested/value.bin"))

    linked = tmp_path / "result/nested/value.bin"
    assert linked.read_bytes() == b"value"
    assert linked.stat().st_ino == source.stat().st_ino


def test_link_accepts_text_and_pure_paths(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"value")

    def build(draft: Draft) -> None:
        draft.link(str(source), PurePosixPath("a/value.bin"))
        draft.link(source, "b/value.bin")

    publish(tmp_path / "result", build)

    assert (tmp_path / "result/a/value.bin").read_bytes() == b"value"
    assert (tmp_path / "result/b/value.bin").read_bytes() == b"value"


def test_hard_linked_result_aliases_its_source_inode(tmp_path: Path) -> None:
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_bytes(b"epoch-10")
    publish(tmp_path / "release", lambda draft: draft.link(checkpoint, "model.ckpt"))

    with checkpoint.open("r+b") as stream:  # an in-place rewrite of the same inode
        stream.write(b"epoch-11")

    assert (tmp_path / "release/model.ckpt").read_bytes() == b"epoch-11"


@pytest.mark.parametrize(
    "unsafe", ["", ".", "..", "../escape", "/absolute", "a//b", "a/./b", "safe\0truncated"]
)
def test_link_rejects_unsafe_draft_paths(tmp_path: Path, unsafe: str) -> None:
    source = tmp_path / "source"
    source.write_text("value")

    with pytest.raises(ConfigurationError, match="safe and relative|embedded NUL"):
        publish(tmp_path / "result", lambda draft: draft.link(source, unsafe))

    assert not (tmp_path / "result").exists()


def test_link_rejects_nul_source_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="embedded NUL"):
        publish(tmp_path / "result", lambda draft: draft.link(f"{tmp_path}/a\0b", "a"))


def test_link_rejects_occupied_draft_path(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_text("value")

    def build(draft: Draft) -> None:
        (draft.path / "value").write_text("occupied")
        draft.link(source, "value")

    with pytest.raises(DestinationExists, match="draft path already exists"):
        publish(tmp_path / "result", build)


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory"])
def test_link_rejects_unsafe_source_without_leaving_an_entry(tmp_path: Path, kind: str) -> None:
    source = tmp_path / "source"
    if kind == "symlink":
        (tmp_path / "target").write_text("value")
        source.symlink_to(tmp_path / "target")
    elif kind == "fifo":
        os.mkfifo(source)
    elif kind == "directory":
        source.mkdir()

    def build(draft: Draft) -> None:
        with pytest.raises(UnsafeFilesystem, match="hard-link source"):
            draft.link(source, "unsafe")
        assert not os.path.lexists(draft.path / "unsafe")
        (draft.path / "safe").write_text("value")

    publish(tmp_path / "result", build)

    assert sorted(path.name for path in (tmp_path / "result").iterdir()) == ["safe"]


def test_link_tree_mirrors_a_checkpoint_directory(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "shards").mkdir(parents=True)
    for index in range(3):
        (checkpoint / "shards" / f"model-{index}.safetensors").write_bytes(b"%d" % index)
    (checkpoint / "config.json").write_text("{}")

    def build(draft: Draft) -> None:
        draft.link_tree(checkpoint, "model")
        draft.link_tree(str(checkpoint / "shards"))

    publish(tmp_path / "release", build)

    release = tmp_path / "release"
    assert sorted(path.name for path in release.iterdir()) == [
        "model",
        "model-0.safetensors",
        "model-1.safetensors",
        "model-2.safetensors",
    ]
    assert (release / "model/config.json").stat().st_ino == (
        checkpoint / "config.json"
    ).stat().st_ino
    assert (release / "model/shards/model-2.safetensors").read_bytes() == b"2"


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_link_tree_rejects_symlinks_and_special_files(tmp_path: Path, kind: str) -> None:
    tree = tmp_path / "tree"
    (tree / "nested").mkdir(parents=True)
    (tree / "nested/value").write_text("value")
    if kind == "symlink":
        (tree / "nested/alias").symlink_to("value")
    else:
        os.mkfifo(tree / "nested/pipe")

    with pytest.raises(UnsafeFilesystem, match="symlink or special file"):
        publish(tmp_path / "result", lambda draft: draft.link_tree(tree))

    assert not (tmp_path / "result").exists()


def test_link_tree_rejects_symlinked_source_and_occupied_paths(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "value").write_text("value")
    (tmp_path / "alias").symlink_to(tree, target_is_directory=True)

    with pytest.raises(UnsafeFilesystem, match="not a safe directory"):
        publish(tmp_path / "first", lambda draft: draft.link_tree(tmp_path / "alias"))

    def occupied(draft: Draft) -> None:
        (draft.path / "value").write_text("occupied")
        draft.link_tree(tree)

    with pytest.raises(DestinationExists, match="draft path already exists"):
        publish(tmp_path / "second", occupied)


@pytest.mark.parametrize("operation", ["link", "link_tree", "path"])
def test_draft_is_invalid_after_builder_returns(tmp_path: Path, operation: str) -> None:
    source = tmp_path / "source"
    source.write_text("value")
    kept: list[Draft] = []

    publish(tmp_path / "result", kept.append)
    (tmp_path / "victim").mkdir()
    descriptor = os.open(tmp_path / "victim", os.O_RDONLY | os.O_DIRECTORY)
    try:
        draft: Any = kept[0]
        with pytest.raises(RuntimeError, match="no longer valid"):
            if operation == "link":
                draft.link(source, "injected")
            elif operation == "link_tree":
                draft.link_tree(tmp_path / "victim")
            else:
                _ = draft.path
    finally:
        os.close(descriptor)

    assert list((tmp_path / "victim").iterdir()) == []
    assert list((tmp_path / "result").iterdir()) == []


def test_draft_is_invalid_after_builder_fails(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_text("value")
    kept: list[Draft] = []

    def fail(draft: Draft) -> None:
        kept.append(draft)
        raise ValueError("stop")

    with pytest.raises(ValueError, match="stop"):
        publish(tmp_path / "result", fail)
    with pytest.raises(RuntimeError, match="no longer valid"):
        kept[0].link(source, "late")


def test_link_reports_a_missing_source_as_a_configuration_error(tmp_path: Path) -> None:
    def build(draft: Draft) -> None:
        with pytest.raises(ConfigurationError, match="hard-link source does not exist"):
            draft.link(tmp_path / "missing", "value")
        with pytest.raises(ConfigurationError, match="does not exist"):
            draft.link_tree(tmp_path / "missing")

    publish(tmp_path / "result", build)

    assert list((tmp_path / "result").iterdir()) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_link_reports_an_inaccessible_source_as_a_configuration_error(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "weights.bin").write_bytes(b"weights")
    locked.chmod(0o000)

    def build(draft: Draft) -> None:
        with pytest.raises(ConfigurationError, match="hard-link source is not accessible"):
            draft.link(locked / "weights.bin", "weights.bin")

    try:
        publish(tmp_path / "result", build)
    finally:
        locked.chmod(0o700)
    assert list((tmp_path / "result").iterdir()) == []


@pytest.mark.parametrize("case", ["contains-draft", "is-draft", "contains-destination"])
def test_link_tree_rejects_a_source_containing_its_draft(tmp_path: Path, case: str) -> None:
    # Regression: linking a tree that contains the stage recursed into its own output.
    (tmp_path / "data").mkdir()
    (tmp_path / "data/value").write_text("x")

    def build(draft: Draft) -> None:
        if case == "contains-draft":
            draft.link_tree(tmp_path, "all")
        elif case == "is-draft":
            draft.link_tree(draft.path, "copy")
        else:
            (draft.path / "outer").mkdir()
            draft.link_tree(draft.path / "outer", "outer/inner")

    with pytest.raises(ConfigurationError, match="contains the draft"):
        publish(tmp_path / "snapshot", build)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["data"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")
def test_linked_file_unreadable_by_its_owner_is_a_configuration_error(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_text("secret")
    source.chmod(0o200)

    with pytest.raises(ConfigurationError, match="not readable by its owner: value"):
        publish(tmp_path / "result", lambda draft: draft.link(source, "value"))

    assert sorted(path.name for path in tmp_path.iterdir()) == ["source"]
