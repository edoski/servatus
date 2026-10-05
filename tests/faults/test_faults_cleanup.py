from __future__ import annotations

import os
import stat
import threading
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from servatus.errors import DestinationExists, UnsafeFilesystem
from servatus.publication import Draft, Workspace, publish, publish_file

Sync = Callable[[int], None]


def hidden(parent: Path) -> list[Path]:
    return sorted(parent.glob(".servatus-*"))


def fail_rmdir_of(syscalls: Any, predicate: Callable[[str], bool], times: int = 1) -> list[str]:
    failures: list[str] = []

    def rmdir(real: Callable[..., None], path: str, *args: Any, **kwargs: Any) -> None:
        if len(failures) < times and predicate(str(path)):
            failures.append(str(path))
            raise OSError("injected cleanup failure")
        real(path, *args, **kwargs)

    syscalls.os("rmdir", rmdir)
    return failures


def test_workspace_cleanup_failure_is_reported_then_reclaimed(
    tmp_path: Path, syscalls: Any
) -> None:
    destination = tmp_path / "model"
    failures = fail_rmdir_of(syscalls, lambda name: name.endswith(".work"))

    with Workspace(destination, identity=b"run") as workspace:
        (workspace.path / "last.ckpt").write_bytes(b"weights")
        with pytest.warns(RuntimeWarning, match="private cleanup remains pending"):
            publication = workspace.publish(
                lambda draft: draft.link(workspace.path / "last.ckpt", "model.bin")
            )

    assert failures
    assert publication.cleanup_pending is True
    assert (destination / "model.bin").read_bytes() == b"weights"
    assert len(hidden(tmp_path)) == 1

    # A requeued task re-entering after success reclaims the residue, then learns it is done.
    with (
        pytest.raises(DestinationExists, match="already exists"),
        Workspace(destination, identity=b"run"),
    ):
        pytest.fail("redundant work was exposed")
    assert hidden(tmp_path) == []


def test_workspace_discard_failure_propagates(tmp_path: Path, syscalls: Any) -> None:
    fail_rmdir_of(syscalls, lambda name: name.endswith(".work"))

    with Workspace(tmp_path / "result", identity=b"run") as workspace:
        with pytest.raises(OSError, match="injected cleanup failure"):
            workspace.discard()
        with pytest.raises(RuntimeError, match="already discarded"):
            workspace.publish(lambda draft: None)


def test_retirement_cleanup_failure_is_nonfatal(tmp_path: Path, syscalls: Any) -> None:
    source = tmp_path / "bundle"
    source.mkdir(mode=0o700)
    fail_rmdir_of(syscalls, lambda name: name == "bundle")

    with pytest.warns(RuntimeWarning, match="private cleanup remains pending"):
        publication = publish(tmp_path / "result", lambda draft: None, retire=source)

    assert publication.cleanup_pending is True
    assert (tmp_path / "result").is_dir()
    assert source.is_dir()


def test_retirement_parent_sync_failure_is_nonfatal(tmp_path: Path, syscalls: Any) -> None:
    source = tmp_path / "bundle"
    source.mkdir(mode=0o700)
    destination = tmp_path / "result"
    failed: list[bool] = []

    def fail_cleanup_sync(real: Sync, fd: int) -> None:
        if not failed and destination.exists() and not source.exists():
            failed.append(True)
            raise OSError("injected retirement parent sync failure")
        real(fd)

    syscalls.on_sync(fail_cleanup_sync)

    with pytest.warns(RuntimeWarning, match="private cleanup remains pending"):
        publication = publish(destination, lambda draft: None, retire=source)

    assert failed
    assert publication.cleanup_pending is True
    assert destination.is_dir()
    assert not source.exists()


def test_stage_cleanup_failure_keeps_writer_exception(tmp_path: Path, syscalls: Any) -> None:
    fail_rmdir_of(syscalls, lambda name: name.startswith(".servatus-stage-"))
    failure = RuntimeError("invalid output")

    def fail(path: Path) -> None:
        path.write_text("partial")
        raise failure

    with pytest.raises(RuntimeError, match="invalid output") as raised:
        publish_file(tmp_path / "result", fail)

    assert raised.value is failure
    assert any("could not remove the failed stage" in note for note in failure.__notes__)
    assert len(hidden(tmp_path)) == 1


@pytest.mark.usefixtures("linux_fallback")
def test_file_fallback_reports_only_private_cleanup_pending(tmp_path: Path, syscalls: Any) -> None:
    fail_rmdir_of(syscalls, lambda name: name.startswith(".servatus-stage-"))

    with pytest.warns(RuntimeWarning, match="private cleanup remains pending"):
        publication = publish_file(tmp_path / "result", lambda path: path.write_text("complete"))

    assert publication.cleanup_pending is True
    assert publication.destination.read_text() == "complete"
    assert len(hidden(tmp_path)) == 1


def test_cleanup_preserves_root_substituted_during_removal(tmp_path: Path, syscalls: Any) -> None:
    moved = tmp_path / "moved-container"
    substituted: list[Path] = []

    with Workspace(tmp_path / "result", identity=b"request") as workspace:
        container = workspace.path.parent
        container_key = container.stat().st_ino

        def listdir(real: Callable[..., list[str]], target: Any) -> list[str]:
            if (
                not substituted
                and isinstance(target, int)
                and os.fstat(target).st_ino == container_key
            ):
                container.rename(moved)
                container.mkdir(mode=0o700)
                (container / "replacement").write_text("preserve")
                substituted.append(container)
            return real(target)

        syscalls.os("listdir", listdir)
        with pytest.warns(RuntimeWarning, match="private cleanup remains pending"):
            publication = workspace.publish(lambda draft: None)

    assert publication.cleanup_pending is True
    assert moved.is_dir()
    assert (container / "replacement").read_text() == "preserve"


def test_cleanup_preserves_nested_directory_substituted_before_open(
    tmp_path: Path, syscalls: Any
) -> None:
    moved = tmp_path / "moved-nested"
    substituted: list[bool] = []

    with Workspace(tmp_path / "result", identity=b"request") as workspace:
        nested = workspace.path / "nested"
        nested.mkdir()
        (nested / "checkpoint").write_text("preserve")

        def open_(real: Callable[..., int], path: Any, *args: Any, **kwargs: Any) -> int:
            if not substituted and path == "nested" and kwargs.get("dir_fd") is not None:
                nested.rename(moved)
                nested.mkdir(mode=0o700)
                (nested / "replacement").write_text("preserve")
                substituted.append(True)
            return real(path, *args, **kwargs)

        syscalls.os("open", open_)
        with pytest.warns(RuntimeWarning, match="private cleanup remains pending"):
            publication = workspace.publish(lambda draft: None)

    assert publication.cleanup_pending is True
    assert (moved / "checkpoint").read_text() == "preserve"
    assert (nested / "replacement").read_text() == "preserve"


def test_link_cleanup_failure_poisons_the_draft(tmp_path: Path, syscalls: Any) -> None:
    source = tmp_path / "source"
    source.write_text("value")
    inspected: list[bool] = []

    def stat(real: Callable[..., os.stat_result], path: Any, *args: Any, **kwargs: Any) -> Any:
        if not inspected and path == "failed" and kwargs.get("dir_fd") is not None:
            inspected.append(True)
            raise OSError("injected inspection failure")
        return real(path, *args, **kwargs)

    def unlink(real: Callable[..., None], path: Any, *args: Any, **kwargs: Any) -> None:
        if path == "failed" and kwargs.get("dir_fd") is not None:
            raise OSError("injected unlink failure")
        real(path, *args, **kwargs)

    syscalls.os("stat", stat)
    syscalls.os("unlink", unlink)

    def build(draft: Draft) -> None:
        with pytest.raises(UnsafeFilesystem, match="could not be removed"):
            draft.link(source, "failed")
        (draft.path / "safe").write_text("complete")

    with pytest.raises(UnsafeFilesystem, match="could not be removed"):
        publish(tmp_path / "result", build)

    assert source.read_text() == "value"
    assert not (tmp_path / "result").exists()


def test_link_selects_source_at_the_hard_link_operation(tmp_path: Path, syscalls: Any) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"original")
    moved = tmp_path / "moved.bin"

    def link(real: Callable[..., None], *args: Any, **kwargs: Any) -> None:
        source.rename(moved)
        source.write_bytes(b"replacement")
        real(*args, **kwargs)

    syscalls.os("link", link)

    publish(tmp_path / "result", lambda draft: draft.link(source, "value.bin"))

    assert (tmp_path / "result/value.bin").read_bytes() == b"replacement"
    assert moved.read_bytes() == b"original"


def test_link_tree_withdraws_an_entry_substituted_during_linking(
    tmp_path: Path, syscalls: Any
) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "value").write_text("inspected")

    def link(real: Callable[..., None], source: Any, *args: Any, **kwargs: Any) -> None:
        if source == "value":
            (tree / "value").unlink()
            (tree / "value").write_text("substituted")
        real(source, *args, **kwargs)

    syscalls.os("link", link)

    def build(draft: Draft) -> None:
        with pytest.raises(UnsafeFilesystem, match="substituted"):
            draft.link_tree(tree)
        assert list(draft.path.iterdir()) == []

    publish(tmp_path / "result", build)

    assert list((tmp_path / "result").iterdir()) == []


# -- identity-stage cleanup on the Linux fallback --------------------------------------------
def fail_identity_unlink(syscalls: Any) -> None:
    def unlink(real: Callable[..., None], path: Any, *args: Any, **kwargs: Any) -> None:
        if str(path).startswith(".identity-"):
            raise OSError("injected identity stage cleanup failure")
        real(path, *args, **kwargs)

    syscalls.os("unlink", unlink)


@pytest.mark.usefixtures("linux_fallback")
def test_identity_fallback_reports_stage_residue_without_failing(
    tmp_path: Path, syscalls: Any
) -> None:
    fail_identity_unlink(syscalls)

    with (
        pytest.warns(RuntimeWarning, match="identity-stage cleanup"),
        Workspace(tmp_path / "result", identity=b"request") as workspace,
    ):
        container = workspace.path.parent
        assert (container / ".identity").is_file()
        assert len(list(container.glob(".identity-*.tmp"))) == 1


@pytest.mark.usefixtures("linux_fallback")
def test_identity_fallback_warns_when_cleanup_sync_fails(tmp_path: Path, syscalls: Any) -> None:
    failures: list[bool] = []

    def fail_cleanup_sync(real: Sync, fd: int) -> None:
        names: set[str] = set(os.listdir(fd)) if stat.S_ISDIR(os.fstat(fd).st_mode) else set()
        identity_stage = any(name.startswith(".identity-") for name in names)
        if ".identity" in names and not identity_stage and not failures:
            failures.append(True)
            raise OSError("injected identity cleanup sync failure")
        real(fd)

    syscalls.on_sync(fail_cleanup_sync)

    with (
        pytest.warns(RuntimeWarning, match="identity-stage cleanup"),
        Workspace(tmp_path / "result", identity=b"request") as workspace,
    ):
        container = workspace.path.parent
        assert (container / ".identity").is_file()
        assert list(container.glob(".identity-*.tmp")) == []

    assert failures


@pytest.mark.usefixtures("linux_fallback")
def test_identity_warning_preserves_other_thread_warning_policy(
    tmp_path: Path, syscalls: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fail_identity_unlink(syscalls)
    real_warn = warnings.warn
    started = threading.Event()
    resume = threading.Event()
    failures: list[BaseException] = []

    def pause(message: Any, category: Any = None, stacklevel: int = 1, source: Any = None) -> None:
        if "identity-stage cleanup" in str(message):
            started.set()
            if not resume.wait(5):
                raise AssertionError("identity warning did not resume")
        real_warn(message, category, stacklevel, source)

    def initialize() -> None:
        try:
            with Workspace(tmp_path / "result", identity=b"request"):
                pass
        except BaseException as error:
            failures.append(error)

    monkeypatch.setattr(warnings, "warn", pause)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        worker = threading.Thread(target=initialize)
        worker.start()
        try:
            assert started.wait(5)
            with pytest.raises(RuntimeWarning, match="unrelated"):
                real_warn("unrelated", RuntimeWarning)
        finally:
            resume.set()
            worker.join(timeout=5)

    assert not worker.is_alive()
    assert failures == []
