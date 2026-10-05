from __future__ import annotations

import errno
import fcntl
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from servatus.errors import DestinationExists, Unavailable
from servatus.publication import Workspace


@pytest.mark.parametrize("race", ["root", "child", "root-before-child"])
def test_entry_rejects_publication_that_lands_before_the_lease(
    tmp_path: Path, syscalls: Any, race: str
) -> None:
    destination = tmp_path / "result"
    root = Workspace(destination, identity=b"root")
    if race != "root":
        with root:
            pass  # the root work directory must exist to coordinate a child
    late = root if race == "root" else root.child("trial", identity=b"trial")
    canonical = root.path / "trial" if race == "child" else destination
    coordinated = (canonical.parent if race != "root-before-child" else tmp_path).stat()
    published: list[bool] = []

    def flock(real: Callable[[int, int], None], fd: int, operation: int) -> None:
        entry = os.fstat(fd)
        if (
            not published
            and operation == fcntl.LOCK_EX
            and (entry.st_dev, entry.st_ino) == (coordinated.st_dev, coordinated.st_ino)
        ):
            published.append(True)  # a competing worker finishes while this one opens
            other_root = Workspace(destination, identity=b"root")
            other = other_root.child("trial", identity=b"trial") if race == "child" else other_root
            with other:
                other.publish(lambda draft: (draft.path / "value").write_text("canonical"))
        real(fd, operation)

    syscalls.wrap(fcntl, "flock", flock)

    with pytest.raises(DestinationExists, match="already exists"), late:
        pytest.fail("redundant private work was exposed")

    assert published
    assert (canonical / "value").read_text() == "canonical"
    if race != "child":
        # Regression: the late opener used to leave its freshly created container behind.
        assert sorted(path.name for path in tmp_path.iterdir()) == ["result"]
    if race == "child":
        assert sorted(path.name for path in root.path.iterdir()) == ["trial"]


@pytest.mark.parametrize("code", [errno.EBADF, errno.ENOLCK, errno.EOPNOTSUPP, errno.EINVAL])
def test_workspace_coordination_falls_back_to_a_lock_file(
    tmp_path: Path, syscalls: Any, code: int
) -> None:
    # NFS refuses `flock` on a read-only directory descriptor; coordination must still work.
    def flock(real: Callable[[int, int], None], fd: int, operation: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(code, os.strerror(code))
        real(fd, operation)

    syscalls.wrap(fcntl, "flock", flock)
    root = Workspace(tmp_path / "result", identity=b"root")
    with root.child("trial", identity=b"trial") as child:
        (child.path / "value").write_text("x")
        child.publish(lambda draft: draft.link(child.path / "value", "value"))
    with root as workspace:
        workspace.publish(lambda draft: draft.link(root.path / "trial/value", "value"))

    assert (tmp_path / "result/value").read_text() == "x"
    assert sorted(path.name for path in tmp_path.iterdir()) == [".servatus.lock", "result"]
    assert stat.S_IMODE((tmp_path / ".servatus.lock").stat().st_mode) == 0o600


def test_other_directory_lock_failures_are_not_masked(tmp_path: Path, syscalls: Any) -> None:
    def flock(real: Callable[[int, int], None], fd: int, operation: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EIO, os.strerror(errno.EIO))
        real(fd, operation)

    syscalls.wrap(fcntl, "flock", flock)

    with (
        pytest.raises(Unavailable, match="Input/output error"),
        Workspace(tmp_path / "result", identity=b"request"),
    ):
        pass

    assert list(tmp_path.iterdir()) == []
