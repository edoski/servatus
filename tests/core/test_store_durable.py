from __future__ import annotations

import errno
import fcntl
import json
import os
import stat
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from core_helpers import CAMPAIGN_ID, reencode, roster, submit
from support.builders import tasks

from servatus import _fs
from servatus._fs import replace_file
from servatus.campaign import _store
from servatus.campaign._config import Task
from servatus.campaign._state import State, append, decode, encode, seal
from servatus.campaign._store import Store
from servatus.errors import (
    ConfigurationError,
    Conflict,
    CorruptState,
    NotFound,
    Unavailable,
    UnsafeFilesystem,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def plain_fsync(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Behave like a filesystem without ``F_FULLFSYNC`` so every durability request reaches
    ``os.fsync`` exactly once, where these tests observe and inject faults."""
    full_sync: int | None = getattr(fcntl, "F_FULLFSYNC", None)
    real = fcntl.fcntl

    def without_full_sync(descriptor: int, command: int, *args: int) -> object:
        if command == full_sync:
            raise OSError(errno.ENOTSUP, "F_FULLFSYNC disabled by the test")
        return real(descriptor, command, *args)

    if full_sync is not None:
        monkeypatch.setattr(fcntl, "fcntl", without_full_sync)
    yield


def created(tmp_path: Path, state: State | None = None) -> tuple[Path, Store]:
    path = tmp_path / "campaign"
    return path, Store.create(path, state or roster(2, appendable=True))


def leftovers(path: Path) -> list[str]:
    return sorted(name for name in os.listdir(path) if name not in {".lock", "campaign.json"})


# --- create and open -------------------------------------------------------------------------


def test_create_writes_owner_only_canonical_state(tmp_path: Path) -> None:
    state = roster(2, appendable=True)
    path, store = created(tmp_path, state)
    assert store.path == path
    assert stat.S_IMODE(path.stat().st_mode) == 0o700
    assert stat.S_IMODE((path / "campaign.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((path / ".lock").stat().st_mode) == 0o600
    assert (path / "campaign.json").read_bytes() == encode(state)
    assert store.read() == state
    assert Store.open(path).read() == state
    assert Store.open(str(path)).read() == state
    assert leftovers(path) == []


def test_create_refuses_existing_state_and_reuses_an_empty_directory(tmp_path: Path) -> None:
    path, _ = created(tmp_path)
    with pytest.raises(Conflict, match="campaign already exists"):
        Store.create(path, roster(1))
    empty = tmp_path / "empty"
    empty.mkdir(mode=0o700)
    assert Store.create(empty, roster(1)).read() == roster(1)


def test_create_needs_parents_unless_asked(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "campaign"
    with pytest.raises(NotFound, match="parent directory does not exist"):
        Store.create(nested, roster(1))
    assert Store.create(nested, roster(1), parents=True).read() == roster(1)
    assert stat.S_IMODE(nested.stat().st_mode) == 0o700


def test_open_missing_campaigns_is_not_found(tmp_path: Path) -> None:
    with pytest.raises(NotFound, match="campaign does not exist"):
        Store.open(tmp_path / "missing")
    (tmp_path / "bare").mkdir(mode=0o700)
    with pytest.raises(NotFound, match="campaign state does not exist"):
        Store.open(tmp_path / "bare")


@pytest.mark.parametrize("path", ["", "a\0b", "/"])
def test_campaign_paths_must_name_one_directory(path: str) -> None:
    with pytest.raises(ConfigurationError, match="campaign path"):
        Store.open(path)


def test_create_rejects_non_states(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="campaign State"):
        Store.create(tmp_path / "campaign", "state")  # pyright: ignore[reportArgumentType]


# --- unsafe filesystem -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entry", "mode", "message"),
    [
        (".", 0o750, "campaign directory must be owner-only"),
        (".", 0o701, "campaign directory must be owner-only"),
        ("campaign.json", 0o640, "campaign state must be owner-only"),
        (".lock", 0o604, "campaign lock must be owner-only"),
    ],
)
def test_entries_must_be_owner_only(tmp_path: Path, entry: str, mode: int, message: str) -> None:
    path, store = created(tmp_path)
    (path / entry).chmod(mode)
    with pytest.raises(UnsafeFilesystem, match=message):
        Store.open(path)
    with pytest.raises(UnsafeFilesystem, match=message):
        store.read()


def test_foreign_ownership_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, store = created(tmp_path)
    other = os.geteuid() + 1
    monkeypatch.setattr(os, "geteuid", lambda: other)
    with pytest.raises(UnsafeFilesystem, match="owned by the current user"):
        store.read()
    with pytest.raises(UnsafeFilesystem, match="owned by the current user"):
        Store.open(path)


def test_symlinked_campaign_directory_is_rejected(tmp_path: Path) -> None:
    path, _ = created(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(path, target_is_directory=True)
    with pytest.raises(UnsafeFilesystem, match="not a safe directory"):
        Store.open(link)
    with pytest.raises(UnsafeFilesystem, match="not a safe directory"):
        Store.create(link, roster(1))


def test_campaign_path_that_is_a_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "campaign"
    path.write_bytes(b"")
    with pytest.raises(UnsafeFilesystem, match="not a safe directory"):
        Store.create(path, roster(1))
    with pytest.raises(UnsafeFilesystem, match="not a safe directory"):
        Store.open(path)


@pytest.mark.parametrize(
    ("name", "message"),
    [("campaign.json", "not a safe regular file: campaign.json"), (".lock", "lock is not a plain")],
)
def test_symlinked_entries_are_rejected(tmp_path: Path, name: str, message: str) -> None:
    path, store = created(tmp_path)
    target = tmp_path / "elsewhere"
    os.replace(path / name, target)
    (path / name).symlink_to(target)
    with pytest.raises(UnsafeFilesystem, match=message):
        store.read()


def test_non_regular_entries_are_rejected(tmp_path: Path) -> None:
    path, store = created(tmp_path)
    (path / "campaign.json").unlink()
    (path / "campaign.json").mkdir(mode=0o700)
    with pytest.raises(UnsafeFilesystem, match="not a regular file: campaign.json"):
        store.read()
    (path / ".lock").unlink()
    (path / ".lock").mkdir(mode=0o700)
    with pytest.raises(UnsafeFilesystem, match="campaign lock is not a plain file"):
        store.read()


def test_substituted_directory_is_rejected(tmp_path: Path) -> None:
    path, store = created(tmp_path)
    path.rename(tmp_path / "original")
    Store.create(path, roster(2, appendable=True))
    with pytest.raises(UnsafeFilesystem, match="campaign directory was replaced"):
        store.read()
    with pytest.raises(UnsafeFilesystem, match="campaign directory was replaced"):
        store.update(seal)


def test_lock_replaced_while_waiting_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, store = created(tmp_path)
    real = fcntl.flock

    def swap_then_lock(descriptor: int, operation: int) -> None:
        os.replace(path / ".lock", tmp_path / "old-lock")
        (path / ".lock").touch(mode=0o600)
        real(descriptor, operation)

    monkeypatch.setattr(fcntl, "flock", swap_then_lock)
    with pytest.raises(UnsafeFilesystem, match="lock was replaced"):
        store.update(seal)
    monkeypatch.setattr(fcntl, "flock", real)
    assert not store.read().sealed


def test_missing_lock_is_recreated_owner_only(tmp_path: Path) -> None:
    path, store = created(tmp_path)
    (path / ".lock").unlink()
    assert store.read() == roster(2, appendable=True)
    assert stat.S_IMODE((path / ".lock").stat().st_mode) == 0o600


def test_state_removed_after_open_is_not_found(tmp_path: Path) -> None:
    path, store = created(tmp_path)
    (path / "campaign.json").unlink()
    with pytest.raises(NotFound, match="campaign state does not exist"):
        store.read()


# --- corruption and size ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"{", "not valid UTF-8 JSON"),
        (b"[]", "unsupported campaign schema"),
        (b'{"schema_version":7,"schema_version":7}', "duplicate JSON object key"),
    ],
)
def test_corrupt_state_is_reported(tmp_path: Path, content: bytes, message: str) -> None:
    path, store = created(tmp_path)
    (path / "campaign.json").write_bytes(content)
    with pytest.raises(CorruptState, match=message):
        store.read()
    with pytest.raises(CorruptState, match=message):
        Store.open(path)
    with pytest.raises(CorruptState, match=message):
        store.update(seal)
    assert (path / "campaign.json").read_bytes() == content


def test_tampered_invariants_are_corrupt_state(tmp_path: Path) -> None:
    path, store = created(tmp_path, submit(roster(1), ["task-0"])[0])
    raw = json.loads((path / "campaign.json").read_bytes())
    raw["attempts"][0]["intent_revision"] = 2
    (path / "campaign.json").write_bytes(reencode(raw))
    with pytest.raises(CorruptState, match="invalid"):
        store.read()


def test_read_and_write_limits_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, store = created(tmp_path)
    before = (path / "campaign.json").read_bytes()
    monkeypatch.setattr(_store, "MAX_STATE_BYTES", len(before))
    assert Store.open(path).read() == roster(2, appendable=True)
    with pytest.raises(Conflict, match="limit is"):
        store.update(lambda state: append(state, [Task("x")]))
    assert (path / "campaign.json").read_bytes() == before
    assert leftovers(path) == []
    monkeypatch.setattr(_store, "MAX_STATE_BYTES", len(before) - 1)
    with pytest.raises(CorruptState, match="limit is"):
        Store.open(path)


# --- update and cache ------------------------------------------------------------------------


def test_update_commits_and_returns_the_new_state(tmp_path: Path) -> None:
    path, store = created(tmp_path)
    changed = store.update(lambda state: append(state, [Task("x")]))
    assert changed.tasks[-1] == Task("x")
    assert decode((path / "campaign.json").read_bytes()) == changed
    assert Store.open(path).read() == changed


def test_unchanged_update_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, store = created(tmp_path)
    before = (path / "campaign.json").stat()
    writes: list[str] = []

    def record(*_args: object, **_kwargs: object) -> None:
        writes.append("write")

    monkeypatch.setattr(_fs, "replace_file", record)
    state = store.read()
    assert store.update(lambda current: current) is state
    assert store.update(lambda current: append(current, ())) is state
    assert writes == []
    after = (path / "campaign.json").stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


def test_failed_change_leaves_state_untouched(tmp_path: Path) -> None:
    path, store = created(tmp_path)
    before = (path / "campaign.json").read_bytes()

    def broken(state: State) -> State:
        raise Conflict("no")

    with pytest.raises(Conflict, match="no"):
        store.update(broken)
    with pytest.raises(TypeError, match="must return a State"):
        store.update(lambda state: "x")  # pyright: ignore[reportArgumentType, reportUnknownLambdaType]
    assert (path / "campaign.json").read_bytes() == before


def count_decodes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    real = decode

    def counting(data: bytes) -> State:
        calls.append(len(data))
        return real(data)

    monkeypatch.setattr(_store, "decode", counting)
    return calls


def test_content_cache_decodes_only_changed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = created(tmp_path)
    calls = count_decodes(monkeypatch)
    store = Store.open(path)
    assert len(calls) == 1
    first = store.read()
    assert store.read() is first and len(calls) == 1
    changed = store.update(lambda state: append(state, [Task("x")]))
    assert len(calls) == 1, "a commit fills the cache with the bytes it wrote"
    assert store.read() is changed and len(calls) == 1
    other = Store.open(path)
    assert len(calls) == 2
    other.update(seal)
    sealed = store.read()
    assert sealed.sealed and len(calls) == 3
    assert store.read() is sealed and len(calls) == 3
    os.replace(path / "campaign.json", tmp_path / "copy")
    (path / "campaign.json").write_bytes((tmp_path / "copy").read_bytes())
    (path / "campaign.json").chmod(0o600)
    assert store.read() is sealed and len(calls) == 3, "same bytes, new inode: still a hit"


def test_cache_never_masks_corruption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, store = created(tmp_path)
    calls = count_decodes(monkeypatch)
    store.read()
    (path / "campaign.json").write_bytes(b"{}")
    with pytest.raises(CorruptState, match="unsupported campaign schema"):
        store.read()
    assert len(calls) == 1


# --- crash safety ----------------------------------------------------------------------------


def fail(*_args: object, **_kwargs: object) -> None:
    raise OSError("injected failure")


def regular_fsync_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    real = os.fsync

    def fsync(descriptor: int) -> None:
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            fail()
        real(descriptor)

    monkeypatch.setattr(os, "fsync", fsync)


def no_progress(*_args: object) -> int:
    return 0


INJECTIONS: list[Callable[[pytest.MonkeyPatch], None]] = [
    lambda patch: patch.setattr(os, "write", fail),
    lambda patch: patch.setattr(os, "write", no_progress),
    regular_fsync_fails,
    lambda patch: patch.setattr(os, "rename", fail),
]


@pytest.mark.parametrize("inject", INJECTIONS, ids=["write", "no-progress", "file-fsync", "rename"])
def test_precommit_failure_preserves_state_and_removes_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inject: Callable[[pytest.MonkeyPatch], None],
) -> None:
    path, store = created(tmp_path)
    before = (path / "campaign.json").read_bytes()
    inject(monkeypatch)
    with pytest.raises(Unavailable, match="injected failure|no progress"):
        store.update(seal)
    monkeypatch.undo()
    assert (path / "campaign.json").read_bytes() == before
    assert leftovers(path) == []
    assert not store.read().sealed
    assert store.update(seal).sealed


def test_interrupt_before_rename_preserves_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, store = created(tmp_path)
    before = (path / "campaign.json").read_bytes()

    def interrupted(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt("injected interrupt")

    monkeypatch.setattr(os, "rename", interrupted)
    with pytest.raises(KeyboardInterrupt, match="injected interrupt"):
        store.update(seal)
    assert (path / "campaign.json").read_bytes() == before
    assert leftovers(path) == []


def test_directory_sync_failure_after_rename_keeps_new_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, store = created(tmp_path)
    real = os.fsync

    def fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            fail()
        real(descriptor)

    monkeypatch.setattr(os, "fsync", fsync)
    with pytest.raises(Unavailable, match="injected failure"):
        store.update(seal)
    monkeypatch.undo()
    assert leftovers(path) == []
    assert store.read().sealed
    assert Store.open(path).read().sealed


def test_commit_syncs_file_then_directory_before_returning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, store = created(tmp_path)
    events: list[str] = []
    real_fsync, real_rename = os.fsync, os.rename

    def fsync(descriptor: int) -> None:
        events.append("dir-sync" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file-sync")
        real_fsync(descriptor)

    def rename(*args: object, **kwargs: object) -> None:
        events.append("rename")
        real_rename(*args, **kwargs)  # pyright: ignore[reportArgumentType, reportCallIssue]

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "rename", rename)
    store.update(seal)
    assert events == ["file-sync", "rename", "dir-sync"]
    assert leftovers(path) == []


def test_new_directory_is_synced_into_its_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[int] = []
    real = os.fsync

    def fsync(descriptor: int) -> None:
        synced.append(os.fstat(descriptor).st_ino)
        real(descriptor)

    monkeypatch.setattr(os, "fsync", fsync)
    Store.create(tmp_path / "campaign", roster(1))
    assert synced[0] == tmp_path.stat().st_ino
    assert synced[-1] == (tmp_path / "campaign").stat().st_ino


def test_replace_file_never_overwrites_an_existing_stage(tmp_path: Path) -> None:
    directory = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        replace_file(directory, "file", b"one")
        replace_file(directory, "file", b"two", mode=0o640)
    finally:
        os.close(directory)
    assert (tmp_path / "file").read_bytes() == b"two"
    assert stat.S_IMODE((tmp_path / "file").stat().st_mode) == 0o640
    assert os.listdir(tmp_path) == ["file"]


# --- concurrency -----------------------------------------------------------------------------

WORKER = textwrap.dedent(
    """
    import sys
    from servatus.campaign._config import Task
    from servatus.campaign._state import append
    from servatus.campaign._store import Store

    store = Store.open(sys.argv[1])
    for index in range(int(sys.argv[3])):
        name = f"{sys.argv[2]}-{index}"
        store.update(lambda state: append(state, [Task(name)]))
    """
)


def test_concurrent_updates_across_processes_serialize(tmp_path: Path) -> None:
    path = tmp_path / "campaign"
    Store.create(path, roster(1, appendable=True))
    workers, updates = 4, 15
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", WORKER, str(path), f"w{worker}", str(updates)],
            env=environment,
            stderr=subprocess.PIPE,
        )
        for worker in range(workers)
    ]
    for process in processes:
        _, error = process.communicate(timeout=120)
        assert process.returncode == 0, error.decode()
    state = Store.open(path).read()
    assert state.revision == workers * updates
    names = [task.key for task in state.tasks[1:]]
    assert sorted(names) == sorted(f"w{w}-{i}" for w in range(workers) for i in range(updates))
    for worker in range(workers):
        mine = [name for name in names if name.startswith(f"w{worker}-")]
        assert mine == [f"w{worker}-{index}" for index in range(updates)]
    assert leftovers(path) == []


def test_two_handles_in_one_process_see_each_other(tmp_path: Path) -> None:
    path, first = created(tmp_path)
    second = Store.open(path)
    first.update(lambda state: append(state, tasks(3)[2:]))
    assert second.read().tasks == tasks(3)
    second.update(seal)
    assert first.read().sealed
    assert first.read().campaign_id == CAMPAIGN_ID


# --- error mapping and descriptors ------------------------------------------------------------


def open_descriptors() -> int:
    return len(os.listdir("/dev/fd"))


def test_create_closes_every_descriptor_when_a_close_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: an interrupted close of the parent leaked the campaign directory descriptor.
    parents: list[int] = []
    real_open, real_close = os.open, os.close

    def open_(path: object, *args: object, **kwargs: object) -> int:
        fd = real_open(path, *args, **kwargs)  # pyright: ignore[reportArgumentType, reportCallIssue]
        if path == str(tmp_path):
            parents.append(fd)
        return fd

    def close(fd: int) -> None:
        real_close(fd)
        if fd in parents:
            raise KeyboardInterrupt("injected interrupt")

    monkeypatch.setattr(os, "open", open_)
    monkeypatch.setattr(os, "close", close)
    before = open_descriptors()
    with pytest.raises(KeyboardInterrupt, match="injected interrupt"):
        Store.create(tmp_path / "campaign", roster(1))
    monkeypatch.undo()

    assert parents
    assert open_descriptors() == before
    assert Store.open(tmp_path / "campaign").read() == roster(1)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")
def test_unwritable_location_is_a_configuration_error(tmp_path: Path) -> None:
    parent = tmp_path / "read-only"
    parent.mkdir()
    parent.chmod(0o500)
    try:
        with pytest.raises(ConfigurationError, match="not a usable location.*Permission denied"):
            Store.create(parent / "campaign", roster(1))
        with pytest.raises(ConfigurationError, match="not a usable location.*Permission denied"):
            Store.create(parent / "nested" / "campaign", roster(1), parents=True)
    finally:
        parent.chmod(0o700)
    assert list(parent.iterdir()) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")
def test_state_unreadable_by_its_owner_is_a_configuration_error(tmp_path: Path) -> None:
    path, store = created(tmp_path)
    (path / "campaign.json").chmod(0o200)
    with pytest.raises(ConfigurationError, match="not readable by its owner: campaign.json"):
        store.read()


def test_change_os_errors_propagate_unchanged(tmp_path: Path) -> None:
    path, store = created(tmp_path)
    failure = OSError(errno.EIO, "the change itself failed")

    def change(state: State) -> State:
        raise failure

    with pytest.raises(OSError, match="the change itself failed") as raised:
        store.update(change)
    assert raised.value is failure
    assert leftovers(path) == []
