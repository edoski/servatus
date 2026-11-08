"""Test doubles for code that drives Servatus campaigns without a cluster.

``FakeScheduler`` is an in-memory Slurm with a shared log directory. Pass it wherever Servatus
accepts ``connect``: it returns itself as the ``Transport`` and answers exactly the ``sbatch``,
``squeue``, ``scancel``, ``scontrol``, ``find``, ``sh``, and ``tail`` commands Servatus issues,
with the same output formats and the same local command bounds as the real transport. Controls move
jobs through their lifecycle the way Slurm and the batch script would, writing the same run markers,
and inject failures::

    fake = FakeScheduler()
    campaign = Campaign.create(path, tasks, connect=fake)
    job = campaign.submit(campaign.plan(profile)).receipts[0].job
    fake.start(job)      # RUNNING; the batch script writes its start marker
    fake.finish(job)     # every step returns 0; the job settles as COMPLETED in squeue
    fake.purge(job)      # Slurm forgets the job (MinJobAge); its markers stay

Every file carries an mtime from the fake's clock (``clock``, or an internal one that ``advance``
moves), which ``find -printf`` reports, so run time is measured as on a real cluster.

Chains behave as Slurm and their batch scripts would. A job submitted with
``--dependency=afternotok:...`` starts only once every job it names ended without exit 0, and with
``--kill-on-invalid-dep=yes`` Slurm deletes it, unstarted, once any of them exits 0. A later link
runs only the slots whose handoff markers exist. ``signal_limit`` delivers the pre-limit signal,
and a slot of a link that hands off writes the next link's handoff marker exactly when
``servatus.campaign._script.continues`` says, as the rendered shell does::

    fake.start(link1)
    fake.signal_limit(link1)          # the limit marker; the steps get SIGUSR1
    fake.interrupt(link1, "TIMEOUT")  # each running slot hands off to link 2
    fake.start(link2)                 # runs the handed slots

A lease is a job whose batch script serves orders. Steps are ordered into its order directory by
``submit`` and withdrawn by ``cancel`` and ``reconcile`` through the same claim the lease takes,
and controls act as the lease's batch script and Slurm's step scheduler would::

    fake.start(lease)                 # the lease starts; it takes no order yet
    fake.poll(lease)                  # it claims each new order; each step waits for room
    fake.begin_step(step)             # Slurm found room: the step writes its start marker
    fake.end_step(step, 0)            # the step returned: its end marker
    fake.honor_stops(lease)           # every step asked to stop ends (143 by default)
    fake.signal_limit(lease)          # the limit marker; the lease takes no more orders
    fake.interrupt(lease, "TIMEOUT")  # its stop marker; unfinished steps record nothing

Controls and ``count`` name commands by basename (``"sbatch"``, ``"squeue"``, ``"find"``, ...),
except the ``sbatch --version`` connectivity check that precedes every submission, which is
``"ping"``: so ``fail_next("sbatch")`` affects the next real submission, and ``fail_next("ping")``
the check. Commands Servatus never issues raise ``AssertionError`` so that drift fails tests
loudly.
"""

from __future__ import annotations

import fnmatch
import posixpath
import re
import shlex
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath
from typing import cast

from .campaign._config import Resources, Target
from .campaign._remote import MAX_STREAM_BYTES, Completed, Transport
from .campaign._scheduler import (
    ORDER,
    SQUEUE_FORMAT,
    SQUEUE_OPTIONS,
    TAIL,
    WITHDRAW,
    WRITABLE_DIRECTORY,
    find_argv,
)
from .campaign._script import check_command, continues, dependency_options, sbatch_argv
from .errors import ConfigurationError, Unavailable

__all__ = ["FakeJob", "FakeScheduler"]

_RUNNING = "RUNNING"
_PENDING = "PENDING"
_COMPLETED = "COMPLETED"
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_SLOT = re.compile(rb"--job-name=servatus-[0-9a-f]{24}-([0-9]+) ")
_PREVIOUS = re.compile(rb"previous=([0-9]+)")
_HANDED = re.compile(rb"[ \t]*(0|[1-9][0-9]{0,8})[ \t]*\n")


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
    dependencies: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class _Link:
    """What the fake reads from a rendered batch script: the slots it launches, the handoff
    prefixes of this link (``own``, set on gated links) and the next one, the slots that hand
    off, on link 1 each slot's run time before the chain, and a lease's order directory."""

    slots: tuple[int, ...]
    own: str | None = None
    next: str | None = None
    hands_off: frozenset[int] = frozenset()
    previous: Mapping[int, int] = field(default_factory=dict[int, int])
    orders: str | None = None

    @classmethod
    def read(cls, script: bytes) -> _Link:
        slots: list[int] = []
        previous: dict[int, int] = {}
        hands_off: set[int] = set()
        prefixes: dict[str, str] = {}
        pending: int | None = None
        for line in script.split(b"\n"):
            if (found := _PREVIOUS.fullmatch(line)) is not None:
                pending = int(found.group(1))
            elif line.startswith((b"own=", b"next=", b"orders=")):
                name, _, value = line.decode().partition("=")
                prefixes[name] = shlex.split(value)[0]
            elif line.endswith(b"&") and (found := _SLOT.search(line)) is not None:
                slot = int(found.group(1))
                slots.append(slot)
                if pending is not None:
                    previous[slot] = pending
                if f'"$next.{slot}.go"'.encode() in line:
                    hands_off.add(slot)
        return cls(
            tuple(slots),
            prefixes.get("own"),
            prefixes.get("next"),
            frozenset(hands_off),
            previous,
            prefixes.get("orders"),
        )


@dataclass(slots=True)
class _Job:
    job: int
    name: str
    argv: tuple[str, ...]
    script: bytes
    task_count: int
    marker: str
    link: _Link
    relay: bool = False
    after: tuple[int, ...] = ()
    kill_invalid: bool = False
    state: str = _PENDING
    in_queue: bool = True
    started_at: datetime | None = None
    limited: bool = False
    ended: dict[int, int] = field(default_factory=dict[int, int])
    launched: dict[int, int] = field(default_factory=dict[int, int])
    handed_off: set[int] = field(default_factory=set[int])
    taken: dict[str, int | None] = field(default_factory=dict[str, int | None])

    @property
    def lease(self) -> bool:
        return self.link.orders is not None

    @property
    def step_prefix(self) -> str:
        """``<log_root>/`` of this job's steps' markers."""
        return f"{posixpath.dirname(self.marker)}/"

    @property
    def started(self) -> bool:
        return self.started_at is not None

    @property
    def live(self) -> bool:
        return self.state in (_PENDING, _RUNNING)


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
    argv = (*sbatch_argv(target, resources, 1, "0" * 24), *dependency_options((1,)))
    return frozenset(word.partition("=")[0] for word in argv[1:]) | {"--test-only"}


_SBATCH_OPTIONS = _sbatch_options()
_SLURM_CONFIG = b"MinJobAge               = 300 sec\nProctrackType           = proctrack/cgroup\n"
_SBATCH_REQUIRED = ("--parsable", "--no-requeue", "--export", "--ntasks", "--job-name", "--output")


def _lines(rows: Sequence[str]) -> bytes:
    return "".join(f"{row}\n" for row in rows).encode("utf-8")


def _epoch(moment: datetime) -> str:
    """``find -printf %T@``: epoch seconds with GNU find's ten fractional digits."""
    elapsed = moment - _EPOCH
    seconds = elapsed // timedelta(seconds=1)
    micro = (elapsed - timedelta(seconds=seconds)) // timedelta(microseconds=1)
    return f"{seconds}.{micro:06d}0000"


def _exit_code(value: object) -> int:
    if type(value) is not int or not 0 <= value <= 255:
        raise ConfigurationError("exit_code must be an integer from 0 to 255")
    return value


class FakeScheduler:
    """An in-memory Slurm cluster that answers Servatus' scheduler commands.

    Job numbers count up from ``first_job_id``. ``version`` is the ``sbatch --version`` text.
    Every Target reaches the same cluster and the same files, keyed by absolute path. ``clock``
    stamps file mtimes (default: an internal clock at 2030-01-01 UTC that ``advance`` moves).
    """

    def __init__(
        self,
        *,
        first_job_id: int = 1000,
        version: str = "slurm 23.11.4",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if type(first_job_id) is not int or first_job_id < 1:
            raise ConfigurationError("first_job_id must be a positive integer")
        if clock is not None and not callable(clock):
            raise ConfigurationError("clock must be callable")
        self._next_job_id = first_job_id
        self._version = version
        self._now = datetime(2030, 1, 1, tzinfo=UTC)
        self._clock = clock
        self._jobs: list[_Job] = []
        self._files: dict[str, bytes] = {}
        self._mtimes: dict[str, datetime] = {}
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
            "scontrol": self._scontrol,
            "find": self._find_files,
            "sh": self._shell,
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

    def mtime(self, path: str | PurePosixPath) -> datetime:
        """When a file was last written, by the fake's clock."""
        return self._mtimes[str(path)]

    def now(self) -> datetime:
        """The fake's current time, which stamps every file it writes."""
        return self._clock() if self._clock is not None else self._now

    # --- Controls ----------------------------------------------------------------------------

    def start(self, job: int) -> None:
        """The job starts running: its batch script writes the start marker and launches its
        slots, a later chain link only those whose handoff marker exists. A link with no slot to
        launch exits 0 at once. A job whose dependencies are not satisfied cannot start."""
        found = self._find(job)
        if found.state != _PENDING:
            raise ConfigurationError(f"job {found.job} is not pending")
        if not self._eligible(found):
            raise ConfigurationError(f"job {found.job} waits for its dependencies")
        found.state = _RUNNING
        found.started_at = self.now()
        self._write(f"{found.marker}.start")
        if found.lease:
            return
        own = found.link.own
        for slot in found.link.slots:
            if own is None:
                found.launched[slot] = found.link.previous.get(slot, 0)
            elif handed := _HANDED.match(self._files.get(f"{own}.{slot}.go", b"")):
                found.launched[slot] = int(handed.group(1))
        if not found.launched:
            found.state = _COMPLETED
            self._settle()

    def finish_step(self, job: int, slot: int, exit_code: int = 0) -> None:
        """One Task step of a running job returns: its slot's end marker appears, and in a link
        that hands off, the next link's handoff marker when the slot continues."""
        found = self._find(job)
        code = _exit_code(exit_code)
        if found.state != _RUNNING or slot not in found.launched or slot in found.ended:
            raise ConfigurationError(f"job {found.job} has no running step in slot {slot}")
        found.ended[slot] = code
        self._write(f"{found.marker}.{slot}.{code}.end")
        if found.relay and continues(code, interrupted=False, limited=found.limited):
            self._hand_off(found, slot)

    def finish(self, job: int, exit_code: int = 0) -> None:
        """Every remaining step returns ``exit_code`` (a pending job starts first). The job
        settles and stays in ``squeue`` until ``purge``: ``COMPLETED`` when it exits 0, which a
        chain link that hands off does only when no slot handed off, else ``FAILED``."""
        found = self._find(job)
        code = _exit_code(exit_code)
        if found.lease:
            raise ConfigurationError(f"lease {found.job} ends only when Slurm interrupts it")
        if found.state == _PENDING:
            self.start(job)
        if found.state == _COMPLETED and not found.launched:
            return
        if found.state != _RUNNING:
            raise ConfigurationError(f"job {found.job} is not running")
        for slot in found.launched:
            if slot not in found.ended:
                self.finish_step(job, slot, code)
        failed = found.handed_off if found.relay else any(found.ended.values())
        found.state = "FAILED" if failed else _COMPLETED
        self._settle()

    def signal_limit(self, job: int) -> None:
        """Slurm sends the pre-limit signal (``signal_before_end`` before the limit). A chain
        link that hands off gets it in its batch shell, which writes the limit marker and relays
        it; any other job's steps get it from Slurm directly. Every step learns of it."""
        found = self._find(job)
        if found.state != _RUNNING:
            raise ConfigurationError(f"job {found.job} is not running")
        if found.relay and not found.limited:
            self._write(f"{found.marker}.limit")
        found.limited = True

    def poll(self, job: int) -> tuple[str, ...]:
        """A running lease looks at its order directory, as its batch script does every few
        seconds: it claims each order it has not taken, unless the client's claim won, and
        starts its step, which waits for room until ``begin_step``. After the pre-limit signal
        it takes nothing. Returns the steps it started, in name order."""
        found = self._find(job)
        if not found.lease or found.state != _RUNNING:
            raise ConfigurationError(f"job {found.job} is not a running lease")
        if found.limited:
            return ()
        directory = cast(str, found.link.orders)
        taken: list[str] = []
        for path in sorted(self._files):
            step = posixpath.basename(path).removesuffix(".run")
            if posixpath.dirname(path) != directory or not path.endswith(".run"):
                continue
            claim = f"{directory}/{step}.claim"
            if claim not in self._files:
                self._write(claim, b"taken")
            if self._files[claim] == b"taken" and step not in found.taken:
                found.taken[step] = None
                taken.append(step)
        return tuple(taken)

    def begin_step(self, step: str) -> None:
        """Slurm found room for a step its lease started: the step writes its start marker."""
        found = self._lease_of(step)
        if f"{found.step_prefix}{step}.{found.job}.start" in self._files:
            raise ConfigurationError(f"step {step} already began")
        self._write(f"{found.step_prefix}{step}.{found.job}.start")

    def end_step(self, step: str, exit_code: int = 0) -> None:
        """A step returns, begun or still waiting for room: its end marker appears."""
        found = self._lease_of(step)
        code = _exit_code(exit_code)
        found.taken[step] = code
        self._write(f"{found.step_prefix}{step}.{found.job}.0.{code}.end")

    def honor_stops(self, job: int, exit_code: int = 143) -> tuple[str, ...]:
        """The lease acts on every stop request: each step it took that has not ended and has a
        stop marker returns ``exit_code`` (143 by default, as after SIGTERM), a running one
        after SIGUSR1, one still waiting for room without starting. Returns those steps."""
        found = self._find(job)
        stopped = tuple(
            step
            for step, code in found.taken.items()
            if code is None and f"{found.step_prefix}{step}.{found.job}.stop" in self._files
        )
        for step in stopped:
            self.end_step(step, exit_code)
        return stopped

    def interrupt(self, job: int, state: str = "CANCELLED") -> None:
        """Slurm interrupts the job (cancel, time limit, preemption). A started batch script
        writes its stop marker and its unfinished steps record nothing; in a link that hands
        off, each one continues after the pre-limit signal."""
        self._interrupt(self._find(job), state)

    def lose(self, job: int, state: str = "NODE_FAIL") -> None:
        """The batch shell vanishes without writing anything (SIGKILL, node failure)."""
        found = self._find(job)
        if not found.live:
            raise ConfigurationError(f"job {found.job} already ended")
        found.state = state
        self._settle()

    def purge(self, job: int) -> None:
        """Slurm forgets the job (after ``MinJobAge``); ``squeue`` no longer lists it."""
        self._find(job).in_queue = False

    def write_file(
        self, path: str | PurePosixPath, content: bytes = b"", *, at: datetime | None = None
    ) -> None:
        """Create or replace a file, such as a log ``tail`` reads or a run marker, stamped
        ``at`` (default: now)."""
        self._write(str(path), bytes(content), at)

    def remove_file(self, path: str | PurePosixPath) -> None:
        """Remove a file, for example a marker an NFS client does not see yet."""
        del self._files[str(path)], self._mtimes[str(path)]

    def advance(self, **delta: float) -> None:
        """Move the internal clock forward by ``timedelta(**delta)``."""
        if self._clock is not None:
            raise ConfigurationError("advance the clock this FakeScheduler was given instead")
        step = timedelta(**delta)
        if step < timedelta(0):
            raise ConfigurationError("the clock never runs backwards")
        self._now += step

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

    def _write(self, path: str, content: bytes = b"", at: datetime | None = None) -> None:
        moment = self.now() if at is None else at
        if moment.utcoffset() != timedelta(0):
            raise ConfigurationError("file times must be aware UTC datetimes")
        self._files[path] = content
        self._mtimes[path] = moment

    def _interrupt(self, found: _Job, state: str) -> None:
        if not found.live:
            raise ConfigurationError(f"job {found.job} already ended")
        if found.state == _RUNNING:
            self._write(f"{found.marker}.stop")
            for slot in found.launched:
                stopped = slot not in found.ended
                if (
                    stopped
                    and found.relay
                    and continues(None, interrupted=True, limited=found.limited)
                ):
                    self._hand_off(found, slot)
        found.state = state
        self._settle()

    def _hand_off(self, found: _Job, slot: int) -> None:
        """The slot writes the next link's handoff marker: its run time so far, in seconds."""
        if slot not in found.link.hands_off or found.link.next is None:
            return
        elapsed = (self.now() - cast(datetime, found.started_at)) // timedelta(seconds=1)
        self._write(f"{found.link.next}.{slot}.go", f"{found.launched[slot] + elapsed}\n".encode())
        found.handed_off.add(slot)

    def _eligible(self, found: _Job) -> bool:
        """``afternotok``: every job it follows ended, none of them with exit 0."""
        states = [self._find(job).state for job in found.after]
        return all(state not in (_PENDING, _RUNNING, _COMPLETED) for state in states)

    def _settle(self) -> None:
        """``--kill-on-invalid-dep=yes``: Slurm deletes each pending job that follows a job that
        exited 0, before it ever starts."""
        changed = True
        while changed:
            changed = False
            for job in self._jobs:
                ended = (self._find(after).state for after in job.after)
                if job.state == _PENDING and job.kill_invalid and _COMPLETED in ended:
                    job.state = "CANCELLED"
                    changed = True

    def _lease_of(self, step: str) -> _Job:
        for job in self._jobs:
            if step in job.taken and job.taken[step] is None and job.state == _RUNNING:
                return job
        raise ConfigurationError(f"no running lease runs step {step}")

    def _find(self, job: int) -> _Job:
        for candidate in reversed(self._jobs):
            if candidate.job == job:
                return candidate
        raise ConfigurationError(f"FakeScheduler has no job {job}")

    @staticmethod
    def _snapshot(job: _Job) -> FakeJob:
        return FakeJob(
            job.job,
            job.name,
            job.argv,
            job.script,
            job.task_count,
            job.state,
            job.in_queue,
            job.after,
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
        if options.get("--kill-on-invalid-dep", "yes") != "yes":
            raise AssertionError(f"FakeScheduler does not understand {argv!r}")
        dependency = options.get("--dependency")
        after: tuple[int, ...] = ()
        if dependency is not None:
            kind, _, jobs = dependency.partition(":")
            if kind != "afternotok" or not jobs:
                raise AssertionError(f"FakeScheduler does not understand {argv!r}")
            after = tuple(int(job) for job in jobs.split(":"))
            if not all(
                any(item.job == job and item.in_queue for item in self._jobs) for job in after
            ):
                reason = b"sbatch: error: Batch job submission failed: Job dependency problem\n"
                return Completed(1, b"", reason)
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
            _Link.read(stdin),
            relay=(options.get("--signal") or "").startswith("B:"),
            after=after,
            kill_invalid="--kill-on-invalid-dep" in options,
        )
        self._jobs.append(job)
        self._next_job_id += 1
        self._settle()
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
            if job.in_queue and job.name == name and job.live:
                self._interrupt(job, "CANCELLED")
        return Completed(0, b"", b"")

    def _scontrol(self, argv: tuple[str, ...], _stdin: bytes) -> Completed:
        if argv[1:] != ("show", "config"):
            raise AssertionError(f"FakeScheduler does not understand {argv!r}")
        return Completed(0, _SLURM_CONFIG, b"")

    def _shell(self, argv: tuple[str, ...], stdin: bytes) -> Completed:
        """The fixed shell programs: the writable-directory check, and writing or withdrawing a
        lease's order."""
        if argv[:-1] == WRITABLE_DIRECTORY:
            return Completed(0, b"", b"")
        if argv[:-2] == ORDER:
            directory, step = argv[-2:]
            self._write(f"{directory}/{step}.run", stdin)
            return Completed(0, b"", b"")
        if argv[:-4] != WITHDRAW:
            raise AssertionError(f"FakeScheduler does not understand {argv!r}")
        directory, step, void, stop = argv[-4:]
        if stop and stop not in self._files:
            self._write(stop)
        claim = f"{directory}/{step}.claim"
        if claim not in self._files:
            self._write(claim, b"withdrawn")
        if self._files[claim] != b"withdrawn":
            return Completed(0, b"taken\n", b"")
        self._write(void)
        return Completed(0, b"withdrawn\n", b"")

    def _find_files(self, argv: tuple[str, ...], _stdin: bytes) -> Completed:
        root, patterns = (argv[2], argv[9:-3:3]) if len(argv) > 12 else ("", ())
        if not patterns or argv != find_argv(PurePosixPath(root), patterns):
            raise AssertionError(f"FakeScheduler does not understand {argv!r}")
        found = sorted(
            path
            for path in self._files
            if posixpath.dirname(path) == root
            and any(fnmatch.fnmatchcase(posixpath.basename(path), item) for item in patterns)
        )
        return Completed(0, _lines([f"{_epoch(self._mtimes[path])} {path}" for path in found]), b"")

    def _tail(self, argv: tuple[str, ...], _stdin: bytes) -> Completed:
        if argv[0] != TAIL or len(argv) != 5 or argv[1] != "-c" or argv[3] != "--":
            raise AssertionError(f"FakeScheduler does not understand {argv!r}")
        path = argv[4]
        if path not in self._files:
            message = f"tail: cannot open '{path}' for reading: No such file or directory"
            return Completed(1, b"", f"{message}\n".encode())
        return Completed(0, self._files[path][-int(argv[2]) :], b"")
