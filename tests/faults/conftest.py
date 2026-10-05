"""Syscall-boundary fault injection for publication.

Faults are injected by wrapping `os.*` and `fcntl.*` functions exactly as `servatus._fs` and the
publication modules call them, never by replacing private helpers.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import sys
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any

import pytest

from servatus import _fs

Hook = Callable[..., Any]


class Syscalls:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._monkeypatch = monkeypatch
        self._sync_hooks: list[Callable[[Callable[[int], None], int], None]] = []
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


@pytest.fixture
def syscalls(monkeypatch: pytest.MonkeyPatch) -> Syscalls:
    return Syscalls(monkeypatch)


@pytest.fixture(autouse=True)
def fresh_native_lookup() -> Iterator[None]:
    _fs.native_noreplace.cache_clear()
    yield
    _fs.native_noreplace.cache_clear()


@pytest.fixture
def linux_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run as Linux with a libc lacking `renameat2` and `syscall` (treated like ENOSYS)."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", _libc_without_symbols)
    _fs.native_noreplace.cache_clear()


def _libc_without_symbols(*args: object, **kwargs: object) -> object:
    return object()
