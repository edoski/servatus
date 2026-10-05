from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import pytest

from servatus import _fs
from servatus.errors import ConfigurationError, CrossDeviceError, UnsafeFilesystem


def mode_of(path: Path) -> int:
    return stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)


def test_write_new_creates_exact_mode_and_refuses_existing(tmp_path: Path) -> None:
    previous = os.umask(0o077)
    try:
        with _fs.open_path(tmp_path) as directory:
            entry = _fs.write_new(directory.fd, "state", b"payload", mode=0o640)
            with pytest.raises(FileExistsError, match="exists"):
                _fs.write_new(directory.fd, "state", b"other")
    finally:
        os.umask(previous)

    assert (tmp_path / "state").read_bytes() == b"payload"
    assert mode_of(tmp_path / "state") == 0o640
    assert (entry.st_ino, entry.st_size) == ((tmp_path / "state").stat().st_ino, 7)


def test_write_new_writes_large_payloads_completely(tmp_path: Path) -> None:
    data = os.urandom(3 * 1024 * 1024 + 7)
    with _fs.open_path(tmp_path) as directory:
        _fs.write_new(directory.fd, "large", data)

    assert (tmp_path / "large").read_bytes() == data
    assert mode_of(tmp_path / "large") == 0o600


def test_replace_file_creates_then_atomically_replaces(tmp_path: Path) -> None:
    with _fs.open_path(tmp_path) as directory:
        _fs.replace_file(directory.fd, "campaign.json", b"first")
        first = (tmp_path / "campaign.json").stat().st_ino
        _fs.replace_file(directory.fd, "campaign.json", b"second", mode=0o644)

    assert (tmp_path / "campaign.json").read_bytes() == b"second"
    assert (tmp_path / "campaign.json").stat().st_ino != first
    assert mode_of(tmp_path / "campaign.json") == 0o644
    assert sorted(path.name for path in tmp_path.iterdir()) == ["campaign.json"]


def test_replace_file_replaces_a_symlink_without_following_it(tmp_path: Path) -> None:
    (tmp_path / "target").write_bytes(b"untouched")
    (tmp_path / "state").symlink_to(tmp_path / "target")

    with _fs.open_path(tmp_path) as directory:
        _fs.replace_file(directory.fd, "state", b"new")

    assert (tmp_path / "target").read_bytes() == b"untouched"
    assert not (tmp_path / "state").is_symlink()
    assert (tmp_path / "state").read_bytes() == b"new"


def test_replace_file_onto_a_directory_fails_without_residue(tmp_path: Path) -> None:
    (tmp_path / "state").mkdir()

    with _fs.open_path(tmp_path) as directory, pytest.raises(OSError, match="directory"):
        _fs.replace_file(directory.fd, "state", b"new")

    assert sorted(path.name for path in tmp_path.iterdir()) == ["state"]


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "nul\0"])
def test_writers_reject_unsafe_names(tmp_path: Path, name: str) -> None:
    with _fs.open_path(tmp_path) as directory:
        with pytest.raises(ConfigurationError, match="unsafe filesystem leaf"):
            _fs.write_new(directory.fd, name, b"")
        with pytest.raises(ConfigurationError, match="unsafe filesystem leaf"):
            _fs.replace_file(directory.fd, name, b"")

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mode", [-1, 0o1000, True])
def test_writers_reject_invalid_modes(tmp_path: Path, mode: int) -> None:
    with (
        _fs.open_path(tmp_path) as directory,
        pytest.raises(ConfigurationError, match="mode must be"),
    ):
        _fs.write_new(directory.fd, "state", b"", mode=mode)


def test_pin_closes_once_and_rejects_use_after_close(tmp_path: Path) -> None:
    pin = _fs.open_path(tmp_path)
    fd = pin.fd
    pin.close()
    replacement = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        pin.close()  # must not close a reused descriptor number
        os.fstat(replacement)
        with pytest.raises(RuntimeError, match="already closed"):
            _ = pin.fd
    finally:
        os.close(replacement)
    assert fd >= 0


def test_pin_open_verifies_type_and_identity(tmp_path: Path) -> None:
    (tmp_path / "file").write_text("x")
    (tmp_path / "directory").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "directory", target_is_directory=True)

    with _fs.open_path(tmp_path) as parent:
        with pytest.raises(UnsafeFilesystem, match="not a regular file"):
            parent.open("directory", directory=False)
        with pytest.raises(UnsafeFilesystem, match="not a safe directory"):
            parent.open("link")
        with pytest.raises(UnsafeFilesystem, match="substituted before it was opened"):
            parent.open("directory", expected=parent.present("file"))
        with parent.open("file", directory=False) as file:
            assert stat.S_ISREG(file.entry.st_mode)


def test_probe_umask_reads_the_mask_without_residue(tmp_path: Path) -> None:
    previous = os.umask(0o027)
    try:
        with _fs.open_path(tmp_path) as directory:
            assert _fs.probe_umask(directory) == 0o027
    finally:
        os.umask(previous)

    assert list(tmp_path.iterdir()) == []


def test_present_reports_a_disappeared_entry(tmp_path: Path) -> None:
    (tmp_path / "entry").write_text("x")
    with _fs.open_path(tmp_path) as directory:
        entry = directory.present("entry")
        (tmp_path / "entry").unlink()
        with pytest.raises(UnsafeFilesystem, match="disappeared: entry"):
            directory.present("entry")
        with pytest.raises(UnsafeFilesystem, match="disappeared: entry"):
            directory.expect("entry", entry)


def other_device(entry: os.stat_result) -> os.stat_result:
    fields = list(entry)
    fields[2] = entry.st_dev + 1  # st_dev
    return os.stat_result(fields)


def test_open_rejects_a_child_on_another_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mounted").mkdir()
    mounted = (tmp_path / "mounted").stat().st_ino
    real = os.fstat

    def fstat(fd: int) -> os.stat_result:
        entry = real(fd)
        return other_device(entry) if entry.st_ino == mounted else entry

    monkeypatch.setattr(os, "fstat", fstat)
    with _fs.open_path(tmp_path) as directory:
        before = len(os.listdir("/dev/fd"))
        with pytest.raises(CrossDeviceError, match="filesystem boundary: mounted"):
            directory.open("mounted")
        assert len(os.listdir("/dev/fd")) == before


def test_walk_rejects_an_entry_on_another_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mounted").mkdir()
    real = os.stat

    def stat_(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        entry = real(path, *args, **kwargs)
        return other_device(entry) if path == "mounted" else entry

    monkeypatch.setattr(os, "stat", stat_)
    visited: list[str] = []
    with (
        _fs.open_path(tmp_path) as directory,
        pytest.raises(CrossDeviceError, match="filesystem boundary: mounted"),
    ):
        _fs.walk(directory, lambda parent, name, entry: visited.append(name))
    assert visited == []


def test_replace_file_never_installs_or_deletes_a_substituted_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "state").write_bytes(b"original")
    real = os.stat
    impostors: list[str] = []

    def stat_(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        if isinstance(path, str) and path.startswith(".servatus-replace-") and not impostors:
            os.rename(tmp_path / path, tmp_path / "moved")
            (tmp_path / path).write_bytes(b"impostor")
            impostors.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat_)
    with (
        _fs.open_path(tmp_path) as directory,
        pytest.raises(UnsafeFilesystem, match="replacement stage was substituted"),
    ):
        _fs.replace_file(directory.fd, "state", b"new")

    assert (tmp_path / "state").read_bytes() == b"original"
    assert (
        tmp_path / impostors[0]
    ).read_bytes() == b"impostor"  # cleanup kept what it did not make
    assert (tmp_path / "moved").read_bytes() == b"new"


def test_write_new_cleanup_never_deletes_a_substituted_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def write(fd: int, data: Any) -> int:
        os.rename(tmp_path / "state", tmp_path / "moved")
        (tmp_path / "state").write_bytes(b"impostor")
        raise OSError("injected write failure")

    monkeypatch.setattr(os, "write", write)
    with (
        _fs.open_path(tmp_path) as directory,
        pytest.raises(OSError, match="injected write failure"),
    ):
        _fs.write_new(directory.fd, "state", b"payload")
    monkeypatch.undo()

    assert (tmp_path / "state").read_bytes() == b"impostor"
    assert (tmp_path / "moved").exists()
