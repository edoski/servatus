from __future__ import annotations

import ctypes
import sys
from collections.abc import Iterator

import pytest

from servatus import _fs


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
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", _libc_without_symbols)
    _fs.native_noreplace.cache_clear()


def _libc_without_symbols(*args: object, **kwargs: object) -> object:
    return object()
