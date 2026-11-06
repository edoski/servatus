"""POSIX helpers and syscall-boundary fault injection for publication tests.

Faults are injected by wrapping `os.*` and `fcntl.*` functions exactly as `servatus._fs` and the
publication modules call them, never by replacing private helpers.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import stat
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from servatus import _fs

Hook = Callable[..., Any]


def hidden(parent: Path) -> list[Path]:
    return sorted(parent.glob(".servatus-*"))


def mode_of(path: Path) -> int:
    return stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)


def open_descriptors() -> int:
    return len(os.listdir("/dev/fd"))


def key(entry: os.stat_result) -> tuple[int, int]:
    return entry.st_dev, entry.st_ino


class Syscalls:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._monkeypatch = monkeypatch
        full_sync = getattr(fcntl, "F_FULLFSYNC", None)
        if full_sync is not None:
            # Behave like a filesystem without F_FULLFSYNC so every durability request reaches
            # os.fsync exactly once and sync hooks see one call per sync.
            def reject_full_sync(real: Hook, fd: int, command: int, *args: Any) -> Any:
                if command == full_sync:
                    raise OSError(errno.ENOTSUP, "F_FULLFSYNC disabled by the test")
                return real(fd, command, *args)

            self.wrap(fcntl, "fcntl", reject_full_sync)

    def wrap(self, module: ModuleType, name: str, hook: Hook) -> None:
        """Replace `module.name` with `hook(real, *args, **kwargs)`."""
        real = getattr(module, name)

        def patched(*args: Any, **kwargs: Any) -> Any:
            return hook(real, *args, **kwargs)

        self._monkeypatch.setattr(module, name, patched)

    def os(self, name: str, hook: Hook) -> None:
        self.wrap(os, name, hook)

    def on_sync(self, hook: Callable[[Callable[[int], None], int], None]) -> None:
        """Intercept every durability request as `hook(real_fsync, fd)`."""
        self.os("fsync", hook)


def libc_without_symbols(*args: object, **kwargs: object) -> object:
    return object()


def use_libc(monkeypatch: pytest.MonkeyPatch, system: str, libc: object) -> None:
    def load(*args: object, **kwargs: object) -> object:
        return libc

    monkeypatch.setattr(sys, "platform", system)
    monkeypatch.setattr(ctypes, "CDLL", load)
    _fs.native_noreplace.cache_clear()


@pytest.fixture
def syscalls(monkeypatch: pytest.MonkeyPatch) -> Syscalls:
    return Syscalls(monkeypatch)


@pytest.fixture(autouse=True)
def fresh_native_lookup() -> Iterator[None]:
    """Each test resolves the native no-replace rename from the current (possibly patched) libc."""
    _fs.native_noreplace.cache_clear()
    yield
    _fs.native_noreplace.cache_clear()


@pytest.fixture
def linux_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run as Linux with a libc that exports neither `renameat2` nor `syscall`.

    A missing symbol is treated like ENOSYS, so every commit takes the documented fallback. On
    macOS this exercises the Linux paths without ever issuing a Linux syscall number.
    """
    use_libc(monkeypatch, "linux", object())
