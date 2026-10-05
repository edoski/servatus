"""Shared synthetic inputs for scheduler tests. Nothing here contacts a real cluster."""

from __future__ import annotations

import posixpath
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from servatus.campaign._evidence import JobRef
from servatus.campaign._remote import MAX_STREAM_BYTES, Completed, check_command
from servatus.campaign._scheduler import AttemptQuery

ALLOCATION = "0123456789abcdef01234567"
IDENTITY = f"servatus-{ALLOCATION}"
INTENT = datetime(2030, 1, 1, 12, 0, 0, tzinfo=UTC)
SUBMIT = "2030-01-01T12:00:05"
WINDOW = ("2030-01-01T11:00:00", "2030-01-01T13:00:00")
MISSING = Completed(1, b"", b"slurm_load_jobs error: Invalid job id specified\n")
SLURM_BIN = "/opt/slurm/bin"


def allocation(index: int) -> str:
    return f"{index:024x}"


def query(
    job_id: int = 42,
    *,
    cluster: str | None = None,
    allocation_id: str = ALLOCATION,
    task_count: int = 1,
    intent_at: datetime = INTENT,
) -> AttemptQuery:
    return AttemptQuery(allocation_id, JobRef(job_id, cluster), intent_at, task_count)


def squeue_row(
    state: str = "RUNNING",
    *,
    job: int | str = 42,
    submit: str = SUBMIT,
    identity: str = IDENTITY,
    name: str | None = None,
    comment: str | None = None,
    reason: str = "None",
    start: str = "N/A",
    end: str = "N/A",
) -> str:
    return (
        f"{job}|{identity if name is None else name}|{identity if comment is None else comment}|"
        f"{submit}|{state}|{reason}|{start}|{end}\n"
    )


def sacct_row(
    state: str = "RUNNING",
    *,
    job: int | str = 42,
    submit: str = SUBMIT,
    identity: str = IDENTITY,
    cluster: str = "",
    name: str | None = None,
    comment: str | None = None,
    exit_code: str = "0:0",
    reason: str = "None",
    start: str = "Unknown",
    end: str = "Unknown",
) -> str:
    return (
        f"{job}|{cluster}|{identity if name is None else name}|"
        f"{identity if comment is None else comment}|{submit}|{state}|{exit_code}|{reason}|"
        f"{start}|{end}\n"
    )


Reply = Completed | BaseException | Callable[[tuple[str, ...]], Completed]


@dataclass
class Scripted:
    """A Transport that answers by command basename from queued replies (FIFO per command).

    A command without queued replies answers with ``default`` (an empty success), so the
    best-effort step query need not be scripted in every test.
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

    def named(self, command: str) -> list[tuple[str, ...]]:
        return [call for call in self.calls if posixpath.basename(call[0]) == command]


def ok(text: str | bytes) -> Completed:
    return Completed(0, text.encode() if isinstance(text, str) else text, b"")


def option(argv: Sequence[str], flag: str) -> str:
    return argv[list(argv).index(flag) + 1]
