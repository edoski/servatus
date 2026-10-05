from __future__ import annotations

import fcntl
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from servatus.errors import DestinationExists
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
