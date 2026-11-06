"""Test doubles for code that drives Servatus campaigns without a cluster.

``FakeScheduler`` is an in-memory Slurm with a shared log directory. Pass it wherever Servatus
accepts ``connect``: it returns itself as the ``Transport`` and answers exactly the ``sbatch``,
``squeue``, ``scancel``, ``find``, and ``tail`` commands Servatus issues, with the same output
formats and the same local command bounds as the real transport. Controls move jobs through their
lifecycle the way Slurm and the batch script would, writing the same run markers, and inject
failures::

    fake = FakeScheduler()
    campaign = Campaign.create(path, tasks, connect=fake)
    job = campaign.submit(campaign.plan(profile)).receipts[0].job
    fake.start(job)      # RUNNING; the batch script writes its start marker
    fake.finish(job)     # every step returns 0; the job settles as COMPLETED in squeue
    fake.purge(job)      # Slurm forgets the job (MinJobAge); its markers stay

Controls and ``count`` name commands by basename (``"sbatch"``, ``"squeue"``, ``"find"``, ...),
except the ``sbatch --version`` connectivity check that precedes every submission, which is
``"ping"``: so ``fail_next("sbatch")`` affects the next real submission, and ``fail_next("ping")``
the check. Commands Servatus never issues raise ``AssertionError`` so that drift fails tests
loudly.
"""

from __future__ import annotations

import fnmatch
import posixpath
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .campaign._config import Resources, Target
from .campaign._remote import MAX_STREAM_BYTES, Completed, Transport
from .campaign._scheduler import SQUEUE_FORMAT, SQUEUE_OPTIONS, TAIL, find_argv
from .campaign._script import check_command, sbatch_argv
from .errors import ConfigurationError, Unavailable

__all__ = ["FakeJob", "FakeScheduler"]

_RUNNING = "RUNNING"
_PENDING = "PENDING"


@dataclass(frozen=True, slots=True)
class FakeJob:
    """A read-only snapshot of one job submitted to a ``FakeScheduler``.

    ``state`` is what ``squeue`` shows; ``in_queue`` is false once Slurm purged the job.
    """

    job: int
    name: str
    argv: tuple[str, ...]
    script: bytes = field(repr=False)
    task_count: int
    state: str
    in_queue: bool


@dataclass(slots=True)
class _Job:
    job: int
    name: str
    argv: tuple[str, ...]
    script: bytes
    task_count: int
    marker: str
    state: str = _PENDING
    in_queue: bool = True
    ended: dict[int, int] = field(default_factory=dict[int, int])

    @property
    def started(self) -> bool:
        return self.state != _PENDING


def _key(argv: Sequence[str]) -> str:
    """A command's control key: its basename, or ``ping`` for ``sbatch --version``."""
    command = posixpath.basename(argv[0])
    return "ping" if command == "sbatch" and tuple(argv[1:]) == ("--version",) else command


def _sbatch_options() -> frozenset[str]:
    """Every option ``sbatch_argv`` can emit, rendered for a request that uses them all."""
    target = Target(
        slurm_bin="/bin",
        work_root="/work",
        log_root="/logs",
        partitions=("p",),
        max_tasks_per_allocation=1,
        max_cpus_per_allocation=1,
        max_memory_mib_per_allocation=1,
        max_time_limit="00:02:00",
        account="a",
        qos="q",
        constraint="c",
        gpu_gres="gpu",
        max_gpus_per_allocation=1,
    )
    resources = Resources(
        cpus=1, memory_mib=1, time_limit="00:02:00", gpus=1, signal_before_end="00:00:30"
    )
    argv = sbatch_argv(target, resources, 1, "0" * 24)
    return frozenset(word.partition("=")[0] for word in argv[1:]) | {"--test-only"}


_SBATCH_OPTIONS = _sbatch_options()
_SBATCH_REQUIRED = ("--parsable", "--no-requeue", "--export", "--ntasks", "--job-name", "--output")


def _lines(rows: Sequence[str]) -> bytes:
    return "".join(f"{row}\n" for row in rows).encode("utf-8")


def _exit_code(value: object) -> int:
    if type(value) is not int or not 0 <= value <= 255:
        raise ConfigurationError("exit_code must be an integer from 0 to 255")
    return value


class FakeScheduler:
    """An in-memory Slurm cluster that answers Servatus' scheduler commands.

    Job numbers count up from ``first_job_id``. ``version`` is the ``sbatch --version`` text.
    Every Target reaches the same cluster and the same files, keyed by absolute path.
    """

    def __init__(self, *, first_job_id: int = 1000, version: str = "slurm 23.11.4") -> None:
        if type(first_job_id) is not int or first_job_id < 1:
            raise ConfigurationError("first_job_id must be a positive integer")
        self._next_job_id = first_job_id
        self._version = version
        self._jobs: list[_Job] = []
        self._files: dict[str, bytes] = {}
        self._calls: list[tuple[str, ...]] = []
        self._failures: dict[str, list[Completed | None]] = {}
        self._lost: dict[str, int] = {}

    # --- Connect and Transport ---------------------------------------------------------------

    def __call__(self, target: Target) -> Transport:
        """Act as ``connect``: every target reaches this one fake cluster."""
        if not isinstance(target, Target):
            raise ConfigurationError("connect needs a Target")
        return self

    def run(
        self, argv: Sequence[str], *, stdin: bytes = b"", max_stdout: int = MAX_STREAM_BYTES
    ) -> Completed:
        """Answer one command exactly as the real transport and Slurm would."""
        fields = check_command(argv)
        self._calls.append(fields)
        command = _key(fields)
        if queued := self._failures.get(command):
            reply = queued.pop(0)
            if reply is None:
                raise Unavailable(f"FakeScheduler: injected failure of {command}")
            return reply
        handlers: Mapping[str, Callable[[tuple[str, ...], bytes], Completed]] = {
            "ping": self._ping,
            "sbatch": self._sbatch,
            "squeue": self._squeue,
            "scancel": self._scancel,
            "find": self._find_files,
            "tail": self._tail,
        }
        if command not in handlers:
            raise AssertionError(f"FakeScheduler does not emulate {fields[0]}")
        reply = handlers[command](fields, stdin)
        if self._lost.get(command):
            self._lost[command] -= 1
            raise Unavailable(f"FakeScheduler: reply from {command} was lost")
        if len(reply.stdout) > max_stdout:
            raise Unavailable("command output exceeds its byte bound")
        return reply

    # --- Observation of the fake -------------------------------------------------------------

    @property
    def calls(self) -> tuple[tuple[str, ...], ...]:
        """Every argv received, in order, including injected failures."""
        return tuple(self._calls)

    def count(self, command: str) -> int:
        """How many times a command (by key, such as ``"sbatch"`` or ``"ping"``) was invoked."""
        return sum(1 for call in self._calls if _key(call) == command)

    @property
    def jobs(self) -> tuple[FakeJob, ...]:
        """Snapshots of every submitted job, in submission order."""
        return tuple(self._snapshot(job) for job in self._jobs)

    def job(self, job: int) -> FakeJob:
        """A snapshot of the most recent job with this number."""
        return self._snapshot(self._find(job))

    @property
    def files(self) -> Mapping[str, bytes]:
        """Every file on the fake cluster (logs and run markers) by absolute path."""
        return dict(self._files)

    # --- Controls ----------------------------------------------------------------------------

    def start(self, job: int) -> None:
        """The job starts running: its batch script writes the start marker."""
        found = self._find(job)
        if found.state != _PENDING:
            raise ConfigurationError(f"job {found.job} is not pending")
        found.state = _RUNNING
        self._files[f"{found.marker}.start"] = b""

    def finish_step(self, job: int, slot: int, exit_code: int = 0) -> None:
        """One Task step of a running job returns: its slot's end marker appears."""
        found = self._find(job)
        code = _exit_code(exit_code)
        if found.state != _RUNNING or not 0 <= slot < found.task_count or slot in found.ended:
            raise ConfigurationError(f"job {found.job} has no running step in slot {slot}")
        found.ended[slot] = code
        self._files[f"{found.marker}.{slot}.{code}.end"] = b""

    def finish(self, job: int, exit_code: int = 0) -> None:
        """Every remaining step returns ``exit_code`` (a pending job starts first). The job
        settles as ``COMPLETED`` when every step returned 0, else ``FAILED``, and stays in
        ``squeue`` until ``purge``."""
        found = self._find(job)
        code = _exit_code(exit_code)
        if found.state == _PENDING:
            self.start(job)
        if found.state != _RUNNING:
            raise ConfigurationError(f"job {found.job} is not running")
        for slot in range(found.task_count):
            if slot not in found.ended:
                self.finish_step(job, slot, code)
        found.state = "COMPLETED" if not any(found.ended.values()) else "FAILED"

    def interrupt(self, job: int, state: str = "CANCELLED") -> None:
        """Slurm interrupts the job (cancel, time limit, preemption). A started batch script
        writes its stop marker and its unfinished steps record nothing."""
        self._interrupt(self._find(job), state)

    def lose(self, job: int, state: str = "NODE_FAIL") -> None:
        """The batch shell vanishes without writing anything (SIGKILL, node failure)."""
        found = self._find(job)
        if found.state not in (_PENDING, _RUNNING):
            raise ConfigurationError(f"job {found.job} already ended")
        found.state = state

    def purge(self, job: int) -> None:
        """Slurm forgets the job (after ``MinJobAge``); ``squeue`` no longer lists it."""
        self._find(job).in_queue = False

    def write_file(self, path: str | PurePosixPath, content: bytes = b"") -> None:
        """Create or replace a file, such as a log ``tail`` reads or a run marker."""
        self._files[str(path)] = bytes(content)

    def remove_file(self, path: str | PurePosixPath) -> None:
        """Remove a file, for example a marker an NFS client does not see yet."""
        del self._files[str(path)]

    def fail_next(self, command: str, reply: Completed | None = None) -> None:
        """Make the next ``command`` (by key) fail without acting.

        With no ``reply`` the transport raises ``Unavailable``; otherwise ``reply`` is returned.
        Calls queue up in order.
        """
        self._failures.setdefault(command, []).append(reply)

    def lose_next_reply(self, command: str) -> None:
        """The next ``command`` (by key) takes effect, but its reply is lost (``Unavailable``)."""
        self._lost[command] = self._lost.get(command, 0) + 1

    def set_next_job_id(self, job_id: int) -> None:
        """Choose the next job number, for example to simulate job-number reuse."""
        if type(job_id) is not int or job_id < 1:
            raise ConfigurationError("job_id must be a positive integer")
        self._next_job_id = job_id

    # --- Internals ---------------------------------------------------------------------------

    def _interrupt(self, found: _Job, state: str) -> None:
        if found.state not in (_PENDING, _RUNNING):
            raise ConfigurationError(f"job {found.job} already ended")
        if found.started:
            self._files[f"{found.marker}.stop"] = b""
        found.state = state

    def _find(self, job: int) -> _Job:
        for candidate in reversed(self._jobs):
            if candidate.job == job:
                return candidate
        raise ConfigurationError(f"FakeScheduler has no job {job}")

    @staticmethod
    def _snapshot(job: _Job) -> FakeJob:
        return FakeJob(
            job.job, job.name, job.argv, job.script, job.task_count, job.state, job.in_queue
        )

    @staticmethod
    def _options(argv: tuple[str, ...]) -> dict[str, str | None]:
        options: dict[str, str | None] = {}
        for word in argv[1:]:
            name, equals, value = word.partition("=")
            if not word.startswith("--") or name in options:
                raise AssertionError(f"FakeScheduler does not understand {argv!r}")
            options[name] = value if equals else None
        return options

    def _ping(self, _argv: tuple[str, ...], _stdin: bytes) -> Completed:
        return Completed(0, f"{self._version}\n".encode(), b"")

    def _sbatch(self, argv: tuple[str, ...], stdin: bytes) -> Completed:
        options = self._options(argv)
        if options.keys() - _SBATCH_OPTIONS or any(
            name not in options for name in _SBATCH_REQUIRED
        ):
            raise AssertionError(f"FakeScheduler does not understand {argv!r}")
        if options["--export"] != "NIL" or not stdin.startswith(b"#!/bin/sh\n"):
            raise AssertionError("Servatus batch requests use --export=NIL and a /bin/sh script")
        task_count = int(options["--ntasks"] or "0")
        if "--test-only" in options:
            message = (
                f"sbatch: Job {self._next_job_id} to start at 2030-01-01T00:00:00 using "
                f"{task_count} processors on nodes fake1 in partition {options.get('--partition')}"
            )
            return Completed(0, b"", f"{message}\n".encode())
        name = options["--job-name"] or ""
        log_root = posixpath.dirname(options["--output"] or "")
        job = _Job(
            self._next_job_id,
            name,
            argv,
            stdin,
            task_count,
            f"{log_root}/{name.removeprefix('servatus-')}.{self._next_job_id}",
        )
        self._jobs.append(job)
        self._next_job_id += 1
        return Completed(0, f"{job.job}\n".encode(), b"")

    def _squeue(self, argv: tuple[str, ...], _stdin: bytes) -> Completed:
        if len(argv) != len(SQUEUE_OPTIONS) + 3:
            raise AssertionError(f"FakeScheduler does not understand {argv!r}")
        names = argv[len(SQUEUE_OPTIONS) + 1].removeprefix("--name=")
        expected = (argv[0], *SQUEUE_OPTIONS, f"--name={names}", SQUEUE_FORMAT)
        if argv != expected or not names:
            raise AssertionError(f"FakeScheduler does not understand {argv!r}")
        wanted = set(names.split(","))
        rows = [
            f"{job.job}|{job.name}|{job.state}"
            for job in self._jobs
            if job.in_queue and job.name in wanted
        ]
        return Completed(0, _lines(rows), b"")

    def _scancel(self, argv: tuple[str, ...], _stdin: bytes) -> Completed:
        if len(argv) != 3 or argv[1] != "--me" or not argv[2].startswith("--name="):
            raise AssertionError(f"FakeScheduler does not understand {argv!r}")
        name = argv[2].removeprefix("--name=")
        for job in self._jobs:
            if job.in_queue and job.name == name and job.state in (_PENDING, _RUNNING):
                self._interrupt(job, "CANCELLED")
        return Completed(0, b"", b"")

    def _find_files(self, argv: tuple[str, ...], _stdin: bytes) -> Completed:
        root, patterns = (argv[2], argv[9:-1:3]) if len(argv) > 10 else ("", ())
        if not patterns or argv != find_argv(PurePosixPath(root), patterns):
            raise AssertionError(f"FakeScheduler does not understand {argv!r}")
        found = sorted(
            path
            for path in self._files
            if posixpath.dirname(path) == root
            and any(fnmatch.fnmatchcase(posixpath.basename(path), item) for item in patterns)
        )
        return Completed(0, _lines(found), b"")

    def _tail(self, argv: tuple[str, ...], _stdin: bytes) -> Completed:
        if argv[0] != TAIL or len(argv) != 5 or argv[1] != "-c" or argv[3] != "--":
            raise AssertionError(f"FakeScheduler does not understand {argv!r}")
        path = argv[4]
        if path not in self._files:
            message = f"tail: cannot open '{path}' for reading: No such file or directory"
            return Completed(1, b"", f"{message}\n".encode())
        return Completed(0, self._files[path][-int(argv[2]) :], b"")
