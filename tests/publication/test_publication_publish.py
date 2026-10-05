from __future__ import annotations

import os
import stat
import sys
import tempfile
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest

from servatus.errors import (
    ConfigurationError,
    CrossDeviceError,
    DestinationExists,
    UnsafeFilesystem,
    UnsupportedPlatform,
)
from servatus.publication import Draft, Publication, publish, publish_file


def hidden(parent: Path) -> list[Path]:
    return sorted(parent.glob(".servatus-*"))


def mode_of(path: Path) -> int:
    return stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)


@pytest.fixture
def umask_022() -> Iterator[None]:
    previous = os.umask(0o022)
    yield
    os.umask(previous)


# -- directories -----------------------------------------------------------------------------
def test_publish_exposes_complete_directory_and_never_overwrites(tmp_path: Path) -> None:
    destination = tmp_path / "result"

    publication = publish(destination, lambda draft: (draft.path / "value").write_text("first"))

    assert publication == Publication(destination, cleanup_pending=False)
    assert (destination / "value").read_text() == "first"
    assert hidden(tmp_path) == []
    with pytest.raises(DestinationExists, match="already exists"):
        publish(destination, lambda draft: (draft.path / "value").write_text("second"))
    assert (destination / "value").read_text() == "first"
    assert hidden(tmp_path) == []


@pytest.mark.parametrize("publisher", ["directory", "file"])
def test_existing_destination_is_rejected_before_callback(tmp_path: Path, publisher: str) -> None:
    destination = tmp_path / "result"
    destination.write_text("existing")
    called: list[object] = []

    with pytest.raises(DestinationExists, match="already exists"):
        if publisher == "directory":
            publish(destination, called.append)
        else:
            publish_file(destination, called.append)

    assert called == []
    assert destination.read_text() == "existing"
    assert hidden(tmp_path) == []


def test_leaf_symlink_destination_is_never_followed(tmp_path: Path) -> None:
    (tmp_path / "target").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "target", target_is_directory=True)

    with pytest.raises(DestinationExists, match="already exists"):
        publish(tmp_path / "link", lambda draft: None)

    assert list((tmp_path / "target").iterdir()) == []


def test_symlinked_parent_is_resolved_once(tmp_path: Path) -> None:
    (tmp_path / "scratch").mkdir()
    (tmp_path / "outputs").symlink_to(tmp_path / "scratch", target_is_directory=True)

    publication = publish(tmp_path / "outputs" / "run", lambda draft: None)

    assert publication.destination == tmp_path / "scratch" / "run"
    assert (tmp_path / "scratch" / "run").is_dir()


def test_builder_failure_removes_stage_and_propagates(tmp_path: Path) -> None:
    def fail(draft: Draft) -> None:
        (draft.path / "partial").write_text("not canonical")
        raise ValueError("invalid result")

    with pytest.raises(ValueError, match="invalid result"):
        publish(tmp_path / "result", fail)

    assert not (tmp_path / "result").exists()
    assert hidden(tmp_path) == []


def test_interrupt_in_builder_removes_stage(tmp_path: Path) -> None:
    def interrupt(draft: Draft) -> None:
        (draft.path / "partial").write_bytes(b"x")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        publish(tmp_path / "result", interrupt)

    assert list(tmp_path.iterdir()) == []


def test_destination_created_during_build_wins_without_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "result"

    def race(draft: Draft) -> None:
        (draft.path / "ours").write_text("ours")
        destination.mkdir()
        (destination / "theirs").write_text("theirs")

    with pytest.raises(DestinationExists, match="already exists"):
        publish(destination, race)

    assert sorted(path.name for path in destination.iterdir()) == ["theirs"]
    assert hidden(tmp_path) == []


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_publish_rejects_symlinks_and_special_files(tmp_path: Path, kind: str) -> None:
    def build(draft: Draft) -> None:
        if kind == "symlink":
            (draft.path / "unsafe").symlink_to(tmp_path / "outside")
        else:
            os.mkfifo(draft.path / "unsafe")

    with pytest.raises(UnsafeFilesystem, match="symlink or special file"):
        publish(tmp_path / "result", build)

    assert not (tmp_path / "result").exists()
    assert hidden(tmp_path) == []


def test_stage_path_substitution_is_rejected(tmp_path: Path) -> None:
    moved = tmp_path / "moved-stage"

    def substitute(draft: Draft) -> None:
        draft.path.rename(moved)
        draft.path.symlink_to(moved, target_is_directory=True)

    with pytest.raises(UnsafeFilesystem, match="substituted"):
        publish(tmp_path / "result", substitute)

    assert not (tmp_path / "result").exists()
    assert moved.is_dir()


def test_parent_substitution_during_build_cannot_redirect_commit(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    moved = tmp_path / "moved-parent"
    replacement: list[Path] = []

    def substitute(draft: Draft) -> None:
        (draft.path / "ours").write_text("ours")
        parent.rename(moved)
        parent.mkdir()
        stage = parent / draft.path.name
        stage.mkdir()
        (stage / "theirs").write_text("theirs")
        replacement.append(stage)

    with pytest.raises(UnsafeFilesystem, match="substituted"):
        publish(parent / "result", substitute)

    assert not (parent / "result").exists()
    assert not (moved / "result").exists()
    assert (replacement[0] / "theirs").read_text() == "theirs"


@pytest.mark.parametrize("name", ["/", "..", "nested/.."])
def test_destination_must_name_an_entry(tmp_path: Path, name: str) -> None:
    with pytest.raises(ConfigurationError, match="must name an entry"):
        publish(name if name == "/" else f"{tmp_path}/{name}", lambda draft: None)


def test_nul_destination_is_rejected_before_build(tmp_path: Path) -> None:
    called: list[object] = []

    with pytest.raises(ConfigurationError, match="embedded NUL"):
        publish(f"{tmp_path}/result\0truncated", called.append)

    assert called == []
    assert list(tmp_path.iterdir()) == []


def test_missing_parent_is_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="parent is unavailable"):
        publish_file(tmp_path / "missing" / "result", lambda path: path.write_text("x"))


def test_string_and_pathlike_destinations_are_accepted(tmp_path: Path) -> None:
    publish(str(tmp_path / "text"), lambda draft: None)
    publish_file(os.fspath(tmp_path / "file"), lambda path: path.write_bytes(b""))

    assert (tmp_path / "text").is_dir()
    assert (tmp_path / "file").is_file()


def test_unsupported_platform_fails_before_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[object] = []
    monkeypatch.setattr(sys, "platform", "win32")

    with pytest.raises(UnsupportedPlatform, match="Linux or macOS"):
        publish(tmp_path / "result", called.append)

    assert called == []


# -- modes -----------------------------------------------------------------------------------
@pytest.mark.usefixtures("umask_022")
def test_directory_modes_follow_umask_including_link_intermediates(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"value")

    def build(draft: Draft) -> None:
        (draft.path / "made").mkdir(mode=0o700)
        draft.link(source, "nested/deeper/value.bin")

    publish(tmp_path / "result", build)

    for directory in ("result", "result/made", "result/nested", "result/nested/deeper"):
        assert mode_of(tmp_path / directory) == 0o755
    assert mode_of(tmp_path / "result/nested/deeper/value.bin") == mode_of(source)


def test_private_umask_yields_private_publication(tmp_path: Path) -> None:
    previous = os.umask(0o077)
    try:
        publish(tmp_path / "directory", lambda draft: None)
        publish_file(tmp_path / "file", lambda path: path.write_text("x"))
    finally:
        os.umask(previous)

    assert mode_of(tmp_path / "directory") == 0o700
    assert mode_of(tmp_path / "file") == 0o600


def test_explicit_directory_mode_is_applied_before_commit(tmp_path: Path) -> None:
    observed: list[int] = []

    def build(draft: Draft) -> None:
        observed.append(mode_of(draft.path))
        (draft.path / "child").mkdir()

    publish(tmp_path / "result", build, mode=0o550)

    assert observed == [0o700]
    assert mode_of(tmp_path / "result") == 0o550
    assert mode_of(tmp_path / "result/child") == 0o550


@pytest.mark.usefixtures("umask_022")
def test_file_mode_defaults_from_umask_and_overrides_writer_chmod(tmp_path: Path) -> None:
    def write(path: Path) -> None:
        path.write_text("x")
        path.chmod(0o600)

    publish_file(tmp_path / "default", write)
    publish_file(tmp_path / "explicit", write, mode=0o640)

    assert mode_of(tmp_path / "default") == 0o644
    assert mode_of(tmp_path / "explicit") == 0o640


@pytest.mark.parametrize("mode", [-1, 0o1000, 0o4755, True, 1.5, "0o644"])
def test_invalid_modes_are_rejected_before_callback(tmp_path: Path, mode: object) -> None:
    called: list[object] = []

    with pytest.raises(ConfigurationError, match="mode must be"):
        publish(tmp_path / "result", called.append, mode=mode)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="mode must be"):
        publish_file(tmp_path / "result", called.append, mode=mode)  # type: ignore[arg-type]

    assert called == []
    assert list(tmp_path.iterdir()) == []


# -- files -----------------------------------------------------------------------------------
def test_writer_receives_real_name_inside_private_stage(tmp_path: Path) -> None:
    destination = tmp_path / "result.json"
    seen: list[tuple[Path, bool, int]] = []

    def write(path: Path) -> None:
        seen.append((path, path.exists(), mode_of(path.parent)))
        path.write_text('{"status":"complete"}\n')

    publication = publish_file(destination, write)

    (path, existed, stage_mode), *_ = seen
    assert path.name == "result.json"
    assert path.parent.parent == tmp_path
    assert path.parent.name.startswith(".servatus-stage-")
    assert (existed, stage_mode) == (False, 0o700)
    assert publication == Publication(destination, cleanup_pending=False)
    assert destination.read_text() == '{"status":"complete"}\n'
    assert hidden(tmp_path) == []


def test_suffix_appending_writer_publishes_its_bytes(tmp_path: Path) -> None:
    def np_save(path: Path) -> None:
        # numpy.save and savefig append their suffix when the path lacks it.
        target = path if path.suffix == ".npy" else path.with_name(path.name + ".npy")
        target.write_bytes(b"\x93NUMPY payload")

    publish_file(tmp_path / "embeddings.npy", np_save)

    assert (tmp_path / "embeddings.npy").read_bytes() == b"\x93NUMPY payload"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["embeddings.npy"]


def test_temp_then_rename_writer_is_supported_and_leftovers_discarded(tmp_path: Path) -> None:
    def atomic_writer(path: Path) -> None:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(b"weights")
        os.replace(temporary, path)
        (path.parent / "junk").symlink_to("/etc/passwd")
        (path.parent / "scratch").mkdir()

    publication = publish_file(tmp_path / "metrics.json", atomic_writer)

    assert publication.cleanup_pending is False
    assert (tmp_path / "metrics.json").read_bytes() == b"weights"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["metrics.json"]


@pytest.mark.parametrize("result", ["missing", "directory", "symlink", "fifo"])
def test_writer_must_leave_one_regular_file(tmp_path: Path, result: str) -> None:
    target = tmp_path / "target"
    target.write_text("target")

    def write(path: Path) -> None:
        if result == "directory":
            path.mkdir()
        elif result == "symlink":
            path.symlink_to(target)
        elif result == "fifo":
            os.mkfifo(path)

    with pytest.raises(UnsafeFilesystem, match="not a safe regular file|not a regular file"):
        publish_file(tmp_path / "result", write)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["target"]


def test_writer_hard_link_to_outside_file_is_rejected_without_chmod(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.write_text("shared")
    outside.chmod(0o600)

    with pytest.raises(UnsafeFilesystem, match="one link"):
        publish_file(tmp_path / "result", lambda path: os.link(outside, path), mode=0o644)

    assert mode_of(outside) == 0o600
    assert sorted(path.name for path in tmp_path.iterdir()) == ["outside"]


def test_writer_failure_removes_stage_and_propagates_same_exception(tmp_path: Path) -> None:
    failure = ValueError("invalid result")

    def fail(path: Path) -> None:
        path.write_text("not canonical")
        raise failure

    with pytest.raises(ValueError, match="invalid result") as raised:
        publish_file(tmp_path / "result", fail)

    assert raised.value is failure
    assert list(tmp_path.iterdir()) == []


def test_file_destination_race_never_overwrites(tmp_path: Path) -> None:
    destination = tmp_path / "result"

    def race(path: Path) -> None:
        path.write_text("ours")
        destination.write_text("theirs")

    with pytest.raises(DestinationExists, match="already exists"):
        publish_file(destination, race)

    assert destination.read_text() == "theirs"
    assert hidden(tmp_path) == []


# -- retirement ------------------------------------------------------------------------------
def make_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir(mode=0o700)
    (bundle / "private").write_text("authored")
    return bundle


def test_publish_retires_owner_only_sibling_after_commit(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    (bundle / "nested").mkdir(mode=0o500)

    publication = publish(
        tmp_path / "result",
        lambda draft: (draft.path / "complete").write_text("canonical"),
        retire=bundle,
    )

    assert publication.cleanup_pending is False
    assert (tmp_path / "result/complete").read_text() == "canonical"
    assert not bundle.exists()
    assert hidden(tmp_path) == []


@pytest.mark.parametrize("failure", ["builder", "collision"])
def test_precommit_failure_preserves_retirement_source(tmp_path: Path, failure: str) -> None:
    bundle = make_bundle(tmp_path)
    destination = tmp_path / "result"

    def build(draft: Draft) -> None:
        if failure == "builder":
            raise ValueError("invalid result")
        destination.mkdir()

    expected = ValueError if failure == "builder" else DestinationExists
    with pytest.raises(expected, match="invalid result|already exists"):
        publish(destination, build, retire=bundle)

    assert (bundle / "private").read_text() == "authored"


@pytest.mark.parametrize("kind", ["missing", "file", "symlink", "fifo"])
def test_unsafe_retirement_source_is_rejected_before_builder(tmp_path: Path, kind: str) -> None:
    source = tmp_path / "bundle"
    if kind == "file":
        source.write_text("not a directory")
    elif kind == "symlink":
        (tmp_path / "target").mkdir(mode=0o700)
        source.symlink_to(tmp_path / "target", target_is_directory=True)
    elif kind == "fifo":
        os.mkfifo(source)
    called: list[object] = []

    with pytest.raises(UnsafeFilesystem, match="not a safe directory"):
        publish(tmp_path / "result", called.append, retire=source)

    assert called == []
    assert not (tmp_path / "result").exists()


def test_retirement_must_be_private_distinct_sibling(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    bundle.chmod(0o755)
    called: list[object] = []

    with pytest.raises(UnsafeFilesystem, match="owner-only"):
        publish(tmp_path / "result", called.append, retire=bundle)
    bundle.chmod(0o700)
    (tmp_path / "other").mkdir()
    with pytest.raises(ConfigurationError, match="sibling"):
        publish(tmp_path / "other" / "result", called.append, retire=bundle)
    with pytest.raises(ConfigurationError, match="differ"):
        publish(bundle, called.append, retire=bundle)

    assert called == []


def test_substituted_retirement_source_is_preserved_as_pending_cleanup(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    moved = tmp_path / "moved"

    def build(draft: Draft) -> None:
        bundle.rename(moved)
        bundle.mkdir(mode=0o700)
        (bundle / "replacement").write_text("preserve")

    with pytest.warns(RuntimeWarning, match="private cleanup remains pending"):
        publication = publish(tmp_path / "result", build, retire=bundle)

    assert publication.cleanup_pending is True
    assert (tmp_path / "result").is_dir()
    assert (moved / "private").read_text() == "authored"
    assert (bundle / "replacement").read_text() == "preserve"


def test_newly_permissive_retirement_source_is_preserved(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)

    with pytest.warns(RuntimeWarning, match="private cleanup remains pending"):
        publication = publish(tmp_path / "result", lambda draft: bundle.chmod(0o755), retire=bundle)

    assert publication.cleanup_pending is True
    assert (bundle / "private").read_text() == "authored"


def test_warning_filter_cannot_mask_committed_publication(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        publication = publish(tmp_path / "result", lambda draft: bundle.chmod(0o755), retire=bundle)

    assert publication.cleanup_pending is True
    assert (tmp_path / "result").is_dir()


def test_warning_hook_failure_cannot_mask_committed_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = make_bundle(tmp_path)
    calls: list[object] = []

    def fail_warning(*args: object, **kwargs: object) -> None:
        calls.append(args)
        raise RuntimeError("injected warning hook failure")

    monkeypatch.setattr(warnings, "warn", fail_warning)

    publication = publish(tmp_path / "result", lambda draft: bundle.chmod(0o755), retire=bundle)

    assert len(calls) == 1
    assert publication.cleanup_pending is True


# -- cross-device ----------------------------------------------------------------------------
def test_link_rejects_cross_device_source(tmp_path: Path) -> None:
    shared_memory = Path("/dev/shm")
    if not shared_memory.is_dir() or shared_memory.stat().st_dev == tmp_path.stat().st_dev:
        pytest.skip("no separate synthetic temporary filesystem is available")
    with tempfile.TemporaryDirectory(dir=shared_memory) as directory:
        source = Path(directory) / "source"
        source.write_text("value")
        with pytest.raises(CrossDeviceError, match="another filesystem"):
            publish(tmp_path / "result", lambda draft: draft.link(source, "source"))
