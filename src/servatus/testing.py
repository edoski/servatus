"""Test doubles for code that drives Servatus campaigns without a cluster.

``FakeScheduler`` is an in-memory Slurm. Pass it wherever Servatus accepts ``connect``: it returns
itself as the ``Transport`` and answers exactly the ``sbatch``, ``squeue``, ``sacct``, ``scancel``,
and ``tail`` commands Servatus issues, with the same output formats and the same local command
bounds as the real transport. Controls move jobs through their lifecycle and inject failures::

    fake = FakeScheduler()
    campaign = Campaign.create(path, tasks, connect=fake)
    campaign.submit(campaign.plan(profile))
    fake.start(1000)
    fake.finish(1000)

Commands Servatus never issues raise ``AssertionError`` so that drift fails tests loudly.
"""

from __future__ import annotations

import posixpath
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath

from .campaign._config import Target
from .campaign._evidence import TIMESTAMP_FORMAT, JobRef, normalize_state
from .campaign._remote import MAX_STREAM_BYTES, Completed, Transport, check_command
from .campaign._scheduler import (
    SACCT_ALLOCATION_FORMAT,
    SACCT_IDENTITY_FORMAT,
    SACCT_STEP_FORMAT,
    SQUEUE_FORMAT,
    TAIL,
)
from .errors import ConfigurationError, Unavailable

__all__ = ["FakeJob", "FakeScheduler"]

_MISSING = b"slurm_load_jobs error: Invalid job id specified\n"
_VALUED = frozenset({"--jobs", "--name", "--starttime", "--endtime"})
_LATEST = "9999-12-31T23:59:59"


@dataclass(frozen=True, slots=True)
class FakeJob:
    """A read-only snapshot of one job submitted to a ``FakeScheduler``."""

    job: JobRef
    name: str
    argv: tuple[str, ...]
    script: bytes = field(repr=False)
    task_count: int
    state: str
    incarnations: int


@dataclass(slots=True)
class _Sample:
    state: str = "PENDING"
    exit_code: str = "0:0"
    reason: str = "None"
    started_at: str | None = None
    ended_at: str | None = None


@dataclass(slots=True)
class _Incarnation:
    submitted_at: str
    live: _Sample = field(default_factory=_Sample)
    recorded: _Sample = field(default_factory=_Sample)
    steps: dict[int, _Sample] = field(default_factory=dict[int, _Sample])


@dataclass(slots=True)
class _Job:
    job_id: int
    name: str
    comment: str
    argv: tuple[str, ...]
    script: bytes
    task_count: int
    history: list[_Incarnation]
    in_queue: bool = True
    accounted: bool = True

    @property
    def latest(self) -> _Incarnation:
        return self.history[-1]


def _lines(rows: Sequence[str]) -> bytes:
    return "".join(f"{row}\n" for row in rows).encode("utf-8")


def _shown(value: str | None, absent: str) -> str:
    return absent if value is None else value


class FakeScheduler:
    """An in-memory Slurm cluster that answers Servatus' scheduler commands.

    ``cluster`` names a federation cluster: when set, ``sbatch --parsable`` prints
    ``<job id>;<cluster>`` and accounting rows carry it. Job numbers count up from
    ``first_job_id``. ``clock`` supplies submission, start, and end times (UTC now by default);
    pass the same clock the code under test uses so submissions fall in its intent window.
    """

    def __init__(
        self,
        *,
        cluster: str | None = None,
        first_job_id: int = 1000,
        clock: Callable[[], datetime] | None = None,
        version: str = "slurm 23.11.4",
    ) -> None:
        if cluster is not None:
            JobRef(1, cluster)
        if type(first_job_id) is not int or first_job_id < 1:
            raise ConfigurationError("first_job_id must be a positive integer")
        self._cluster = cluster
        self._next_job_id = first_job_id
        self._clock = clock or (lambda: datetime.now(UTC))
        self._version = version
        self._jobs: list[_Job] = []
        self._logs: dict[str, bytes] = {}
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
        command = posixpath.basename(fields[0])
        if queued := self._failures.get(command):
            reply = queued.pop(0)
            if reply is None:
                raise Unavailable(f"FakeScheduler: injected failure of {command}")
            return reply
        handlers: Mapping[str, Callable[[tuple[str, ...], bytes], Completed]] = {
            "sbatch": self._sbatch,
            "squeue": self._squeue,
            "sacct": self._sacct,
            "scancel": self._scancel,
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
        """How many times a command (by basename, such as ``"sbatch"``) was invoked."""
        return sum(1 for call in self._calls if posixpath.basename(call[0]) == command)

    @property
    def jobs(self) -> tuple[FakeJob, ...]:
        """Snapshots of every submitted job, in submission order."""
        return tuple(self._snapshot(job) for job in self._jobs)

    def job(self, job: int | JobRef) -> FakeJob:
        """A snapshot of the most recent job with this number."""
        return self._snapshot(self._find(job))

    # --- Controls ----------------------------------------------------------------------------

    def start(self, job: int | JobRef, *, accounted: bool = True) -> None:
        """Start the job's current incarnation; every Task step starts running.

        ``accounted=False`` leaves accounting showing the previous state (slurmdbd lag).
        """
        found = self._find(job)
        now = self._now()
        latest = found.latest
        latest.live.state, latest.live.started_at = "RUNNING", now
        latest.steps = {
            slot: _Sample("RUNNING", started_at=now) for slot in range(found.task_count)
        }
        self._record(found, accounted)

    def finish(
        self,
        job: int | JobRef,
        state: str = "COMPLETED",
        *,
        exit_code: str = "0:0",
        in_queue: bool = False,
        accounted: bool = True,
    ) -> None:
        """End the current incarnation in ``state``; unfinished steps end the same way.

        ``in_queue=True`` keeps the finished job visible to ``squeue`` (before ``MinJobAge``);
        ``accounted=False`` leaves accounting showing the previous state (slurmdbd lag).
        """
        found = self._find(job)
        now = self._now()
        latest = found.latest
        latest.live.state, latest.live.exit_code, latest.live.ended_at = state, exit_code, now
        for step in latest.steps.values():
            if not normalize_state(step.state).terminal:
                step.state, step.exit_code, step.ended_at = state, exit_code, now
        found.in_queue = in_queue
        self._record(found, accounted)

    def finish_step(
        self, job: int | JobRef, slot: int, state: str = "COMPLETED", *, exit_code: str = "0:0"
    ) -> None:
        """End one Task step (by zero-based slot) of a running job."""
        found = self._find(job)
        if slot not in found.latest.steps:
            raise ConfigurationError(f"job {found.job_id} has no started step for slot {slot}")
        step = found.latest.steps[slot]
        step.state, step.exit_code, step.ended_at = state, exit_code, self._now()

    def requeue(self, job: int | JobRef, *, state: str = "PENDING") -> None:
        """Requeue the job: a new incarnation with a strictly later submit time."""
        found = self._find(job)
        latest = found.latest
        if not normalize_state(latest.live.state).terminal:
            latest.live.state = "REQUEUED"
            latest.live.ended_at = self._now()
            self._record(found, True)
        moment = self._clock()
        previous = datetime.strptime(latest.submitted_at, TIMESTAMP_FORMAT).replace(tzinfo=UTC)
        moment = max(moment, previous + timedelta(seconds=1))
        incarnation = _Incarnation(moment.astimezone(UTC).strftime(TIMESTAMP_FORMAT))
        incarnation.live.state = incarnation.recorded.state = state
        found.history.append(incarnation)
        found.in_queue = True

    def forget(self, job: int | JobRef) -> None:
        """The job ages out of both ``squeue`` and ``sacct``."""
        found = self._find(job)
        found.in_queue = found.accounted = False

    def fail_next(self, command: str, reply: Completed | None = None) -> None:
        """Make the next ``command`` (basename) fail without acting.

        With no ``reply`` the transport raises ``Unavailable``; otherwise ``reply`` is returned.
        Calls queue up in order.
        """
        self._failures.setdefault(command, []).append(reply)

    def lose_next_reply(self, command: str) -> None:
        """The next ``command`` takes effect, but its reply is lost (``Unavailable``)."""
        self._lost[command] = self._lost.get(command, 0) + 1

    def write_log(self, path: str | PurePosixPath, content: bytes) -> None:
        """Set the content ``tail`` reads from a remote log path."""
        self._logs[str(path)] = bytes(content)

    def set_next_job_id(self, job_id: int) -> None:
        """Choose the next job number, for example to simulate job-number reuse."""
        if type(job_id) is not int or job_id < 1:
            raise ConfigurationError("job_id must be a positive integer")
        self._next_job_id = job_id

    # --- Internals ---------------------------------------------------------------------------

    def _now(self) -> str:
        return self._clock().astimezone(UTC).strftime(TIMESTAMP_FORMAT)

    def _find(self, job: int | JobRef) -> _Job:
        job_id = job.job_id if isinstance(job, JobRef) else job
        for candidate in reversed(self._jobs):
            if candidate.job_id == job_id:
                return candidate
        raise ConfigurationError(f"FakeScheduler has no job {job_id}")

    def _snapshot(self, job: _Job) -> FakeJob:
        return FakeJob(
            JobRef(job.job_id, self._cluster),
            job.name,
            job.argv,
            job.script,
            job.task_count,
            job.latest.live.state,
            len(job.history),
        )

    @staticmethod
    def _record(job: _Job, accounted: bool) -> None:
        if accounted:
            live = job.latest.live
            job.latest.recorded = _Sample(
                live.state, live.exit_code, live.reason, live.started_at, live.ended_at
            )

    def _routed(self, options: Mapping[str, str | None]) -> list[_Job]:
        if "--clusters" in options and options["--clusters"] != self._cluster:
            return []
        return list(self._jobs)

    @staticmethod
    def _options(argv: tuple[str, ...], allowed: frozenset[str]) -> dict[str, str | None]:
        options: dict[str, str | None] = {}
        words = iter(argv[1:])
        operands: list[str] = []
        for word in words:
            name, equals, value = word.partition("=")
            if not word.startswith("--"):
                operands.append(word)
            elif name in _VALUED and not equals:
                options[name] = next(words, None)
            else:
                options[name] = value if equals else None
        unknown = options.keys() - allowed
        if unknown or None in (options.get(name, "") for name in _VALUED):
            raise AssertionError(f"FakeScheduler does not understand {argv!r}")
        if operands:
            options["operands"] = " ".join(operands)
        return options

    def _sbatch(self, argv: tuple[str, ...], stdin: bytes) -> Completed:
        if argv[1:] == ("--version",):
            return Completed(0, f"{self._version}\n".encode(), b"")
        allowed = frozenset(
            {
                "--parsable",
                "--export",
                "--nodes",
                "--ntasks",
                "--cpus-per-task",
                "--mem",
                "--time",
                "--partition",
                "--chdir",
                "--job-name",
                "--comment",
                "--output",
                "--error",
                "--account",
                "--qos",
                "--constraint",
                "--gres",
                "--signal",
                "--test-only",
            }
        )
        options = self._options(argv, allowed)
        required = ("--parsable", "--export", "--ntasks", "--job-name", "--comment")
        if "operands" in options or any(name not in options for name in required):
            raise AssertionError(f"FakeScheduler does not understand {argv!r}")
        if options["--export"] != "NIL" or not stdin.startswith(b"#!/bin/sh\n"):
            raise AssertionError("Servatus batch requests use --export=NIL and a /bin/sh script")
        task_count = int(options["--ntasks"] or "0")
        if "--test-only" in options:
            message = (
                f"sbatch: Job {self._next_job_id} to start at {self._now()} using "
                f"{task_count} processors on nodes fake1 in partition {options.get('--partition')}"
            )
            return Completed(0, b"", f"{message}\n".encode())
        job = _Job(
            self._next_job_id,
            options["--job-name"] or "",
            options["--comment"] or "",
            argv,
            stdin,
            task_count,
            [_Incarnation(self._now())],
        )
        self._jobs.append(job)
        self._next_job_id += 1
        suffix = "" if self._cluster is None else f";{self._cluster}"
        return Completed(0, f"{job.job_id}{suffix}\n".encode(), b"")

    def _squeue(self, argv: tuple[str, ...], _stdin: bytes) -> Completed:
        options = self._options(
            argv, frozenset({"--noheader", "--jobs", "--name", "--local", "--clusters", "--format"})
        )
        jobs = [job for job in self._routed(options) if job.in_queue]
        if options.get("--format") == "%i|%j|%k" and "--name" in options:
            rows = [f"{j.job_id}|{j.name}|{j.comment}" for j in jobs if j.name == options["--name"]]
            return Completed(0, _lines(rows), b"")
        if f"--format={options.get('--format')}" != SQUEUE_FORMAT or "--jobs" not in options:
            raise AssertionError(f"FakeScheduler does not understand {argv!r}")
        wanted = {int(item) for item in (options["--jobs"] or "").split(",")}
        banner = "" if "--clusters" not in options else f"CLUSTER: {options['--clusters']}\n"
        rows: list[str] = []
        for job in jobs:
            if job.job_id in wanted:
                live = job.latest.live
                rows.append(
                    f"{job.job_id}|{job.name}|{job.comment}|{job.latest.submitted_at}|"
                    f"{live.state}|{live.reason}|{_shown(live.started_at, 'N/A')}|"
                    f"{_shown(live.ended_at, 'N/A')}"
                )
        if not rows:
            return Completed(1, banner.encode(), _MISSING)
        return Completed(0, banner.encode() + _lines(rows), b"")

    def _sacct(self, argv: tuple[str, ...], _stdin: bytes) -> Completed:
        options = self._options(
            argv,
            frozenset(
                {
                    "--noheader",
                    "--parsable2",
                    "--allocations",
                    "--duplicates",
                    "--jobs",
                    "--name",
                    "--starttime",
                    "--endtime",
                    "--local",
                    "--clusters",
                    "--format",
                }
            ),
        )
        format_ = f"--format={options.get('--format')}"
        start = options.get("--starttime") or ""
        end = options.get("--endtime") or _LATEST
        wanted = (
            None
            if "--jobs" not in options
            else {int(item) for item in (options["--jobs"] or "").split(",")}
        )
        names = set((options.get("--name") or "").split(","))
        jobs = [
            job
            for job in self._routed(options)
            if job.accounted and (wanted is None or job.job_id in wanted)
        ]
        cluster = self._cluster or ""
        rows: list[str] = []
        if format_ == SACCT_STEP_FORMAT and "--allocations" not in options and wanted:
            for job in jobs:
                latest = job.latest
                rows.append(
                    f"{job.job_id}|{job.name}|{latest.recorded.state}|{latest.recorded.exit_code}"
                )
                if latest.recorded.started_at is not None:
                    rows.append(f"{job.job_id}.batch|batch|{latest.recorded.state}|0:0")
                    rows.append(f"{job.job_id}.extern|extern|{latest.recorded.state}|0:0")
                for slot, step in sorted(latest.steps.items()):
                    rows.append(
                        f"{job.job_id}.{slot}|{job.name}-{slot}|{step.state}|{step.exit_code}"
                    )
            return Completed(0, _lines(rows), b"")
        if "--allocations" not in options or "--duplicates" not in options:
            raise AssertionError(f"FakeScheduler does not understand {argv!r}")
        for job in jobs:
            if job.name not in names:
                continue
            for incarnation in job.history:
                recorded = incarnation.recorded
                ended = recorded.ended_at
                if (ended is not None and ended < start) or incarnation.submitted_at > end:
                    continue
                if format_ == SACCT_ALLOCATION_FORMAT and wanted is not None:
                    rows.append(
                        f"{job.job_id}|{cluster}|{job.name}|{job.comment}|"
                        f"{incarnation.submitted_at}|{recorded.state}|{recorded.exit_code}|"
                        f"{recorded.reason}|{_shown(recorded.started_at, 'Unknown')}|"
                        f"{_shown(ended, 'Unknown')}"
                    )
                elif format_ == SACCT_IDENTITY_FORMAT and wanted is None:
                    rows.append(f"{job.job_id}|{job.name}|{job.comment}|{cluster}")
                else:
                    raise AssertionError(f"FakeScheduler does not understand {argv!r}")
        return Completed(0, _lines(rows), b"")

    def _scancel(self, argv: tuple[str, ...], _stdin: bytes) -> Completed:
        options = self._options(argv, frozenset({"--clusters", "--name"}))
        operand = options.get("operands") or ""
        if "--name" not in options or not operand.isdecimal():
            raise AssertionError(f"FakeScheduler does not understand {argv!r}")
        job_id = int(operand)
        jobs = [job for job in self._routed(options) if job.job_id == job_id and job.accounted]
        if not jobs:
            message = f"scancel: error: Kill job error on job id {job_id}: Invalid job id specified"
            return Completed(1, b"", f"{message}\n".encode())
        job = jobs[-1]
        if job.name != options["--name"]:
            return Completed(0, b"", b"")
        if normalize_state(job.latest.live.state).terminal:
            message = (
                f"scancel: error: Kill job error on job id {job_id}: "
                "Job/step already completing or completed"
            )
            return Completed(1, b"", f"{message}\n".encode())
        self.finish(job_id, "CANCELLED by 1000", exit_code="0:15")
        return Completed(0, b"", b"")

    def _tail(self, argv: tuple[str, ...], _stdin: bytes) -> Completed:
        if argv[0] != TAIL or len(argv) != 5 or argv[1] != "-c" or argv[3] != "--":
            raise AssertionError(f"FakeScheduler does not understand {argv!r}")
        path = argv[4]
        if path not in self._logs:
            message = f"tail: cannot open '{path}' for reading: No such file or directory"
            return Completed(1, b"", f"{message}\n".encode())
        return Completed(0, self._logs[path][-int(argv[2]) :], b"")
