"""Shared synthetic inputs for scheduler tests. Nothing here contacts a real cluster."""

from __future__ import annotations

import posixpath
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from servatus.campaign._remote import MAX_STREAM_BYTES, Completed
from servatus.campaign._script import check_command

ALLOCATION = "0123456789abcdef01234567"
IDENTITY = f"servatus-{ALLOCATION}"
SLURM_BIN = "/opt/slurm/bin"
LOG_ROOT = "/cluster/logs/project"


def allocation(index: int) -> str:
    return f"{index:024x}"


def listing(*names: str, root: str = LOG_ROOT) -> bytes:
    """A ``find LOG_ROOT -maxdepth 1 -type f`` reply listing ``names`` in ``root``."""
    return "".join(f"{root}/{name}\n" for name in names).encode()


Reply = Completed | BaseException | Callable[[tuple[str, ...]], Completed]


@dataclass
class Scripted:
    """A Transport that answers by command basename from queued replies (FIFO per command).

    A command without queued replies answers with ``default`` (an empty success).
    """

    replies: dict[str, list[Reply]] = field(default_factory=dict[str, list[Reply]])
    calls: list[tuple[str, ...]] = field(default_factory=list[tuple[str, ...]])
    stdins: list[bytes] = field(default_factory=list[bytes])
    limits: list[int] = field(default_factory=list[int])
    default: Completed = field(default_factory=lambda: Completed(0, b"", b""))

    def queue(self, command: str, *replies: Reply) -> Scripted:
        self.replies.setdefault(command, []).extend(replies)
        return self

    def run(
        self, argv: Sequence[str], *, stdin: bytes = b"", max_stdout: int = MAX_STREAM_BYTES
    ) -> Completed:
        fields = check_command(argv)
        self.calls.append(fields)
        self.stdins.append(stdin)
        self.limits.append(max_stdout)
        queued = self.replies.get(posixpath.basename(fields[0]))
        reply: Reply = queued.pop(0) if queued else self.default
        if isinstance(reply, BaseException):
            raise reply
        if callable(reply):
            return reply(fields)
        return reply


def ok(text: str | bytes) -> Completed:
    return Completed(0, text.encode() if isinstance(text, str) else text, b"")
