"""Campaign behaviour helpers: a fake cluster, a controllable clock, and transport wrappers.

Everything drives the public ``Campaign`` API. Scheduler facts are stated as ``FakeScheduler``
events; transport faults are injected through the public ``connect`` seam.
"""

from __future__ import annotations

import posixpath
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from support.builders import resources, target

from servatus.campaign import Campaign, Completed, Profile, ResultProbe, Target, Task, Transport
from servatus.errors import Unavailable
from servatus.testing import FakeScheduler

START = datetime(2030, 1, 1, 12, 0, 0, tzinfo=UTC)


class Clock:
    """A settable UTC clock shared by the Campaign and the fake cluster."""

    def __init__(self, now: datetime = START) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


def cpu_profile(label: str = "cpu", **changes: Any) -> Profile:
    """A container-free CPU profile: four 2-CPU Tasks per allocation."""
    values: dict[str, Any] = {
        "container": None,
        "gpu_gres": None,
        "max_gpus_per_allocation": 0,
        "max_cpus_per_allocation": 8,
        "max_memory_mib_per_allocation": 4096,
        "max_time_limit": "1-00:00:00",
    }
    values.update(changes)
    return Profile(
        label,
        target(**values),
        resources(cpus=2, memory_mib=1024, gpus=0, time_limit="01:00:00"),
    )


def jobs(count: int, prefix: str = "task") -> tuple[Task, ...]:
    """Direct-launch Tasks with absolute programs, private stdin, and environment."""
    return tuple(
        Task(
            f"{prefix}-{index}",
            ("/usr/bin/python3", "train.py", str(index)),
            stdin=f"secret-{index}\n".encode(),
            env={"SEED": str(index)},
        )
        for index in range(count)
    )


@dataclass
class Probe:
    """A result probe that records each call and reports ``valid`` keys."""

    valid: set[str] = field(default_factory=set[str])
    calls: list[tuple[str, ...]] = field(default_factory=list[tuple[str, ...]])

    def __call__(self, tasks: Sequence[Task]) -> set[str]:
        keys = tuple(task.key for task in tasks)
        self.calls.append(keys)
        return {key for key in keys if key in self.valid}


Hook = Callable[[tuple[str, ...]], Completed | None]


@dataclass
class Wire:
    """A ``connect`` that routes every target to one fake cluster, with injectable faults.

    ``down`` hosts fail every command (``Unavailable``). Each ``before`` hook runs once, before
    the next matching command (by basename; never ``sbatch --version``) reaches the cluster: it
    may raise, mutate the campaign, or return a reply that replaces the cluster's answer.
    """

    fake: FakeScheduler
    hosts: list[str | None] = field(default_factory=list[str | None])
    down: set[str | None] = field(default_factory=set[str | None])
    before: dict[str, list[Hook]] = field(default_factory=dict[str, list[Hook]])

    def __call__(self, target_value: Target) -> Transport:
        self.hosts.append(target_value.host)
        return _Routed(self, target_value.host)

    def on(self, command: str, hook: Hook) -> None:
        self.before.setdefault(command, []).append(hook)


@dataclass
class _Routed:
    wire: Wire
    host: str | None

    def run(
        self, argv: Sequence[str], *, stdin: bytes = b"", max_stdout: int = 1024 * 1024
    ) -> Completed:
        if self.host in self.wire.down:
            raise Unavailable(f"ssh: connect to host {self.host}: Connection refused")
        command = posixpath.basename(argv[0])
        hooks = self.wire.before.get(command, [])
        if hooks and "--version" not in argv:
            reply = hooks.pop(0)(tuple(argv))
            if reply is not None:
                return reply
        return self.wire.fake.run(argv, stdin=stdin, max_stdout=max_stdout)


@dataclass
class World:
    root: Path
    clock: Clock
    fake: FakeScheduler
    wire: Wire

    @property
    def path(self) -> Path:
        return self.root / "campaign"

    def create(
        self, tasks: Iterable[Task], *, appendable: bool = False, probe: ResultProbe | None = None
    ) -> Campaign:
        return Campaign.create(
            self.path,
            tasks,
            appendable=appendable,
            probe=probe,
            connect=self.wire,
            clock=self.clock,
        )

    def open(self, *, probe: ResultProbe | None = None) -> Campaign:
        return Campaign.open(self.path, probe=probe, connect=self.wire, clock=self.clock)

    def scheduler_calls(self) -> int:
        return len(self.fake.calls)


def make_world(root: Path) -> World:
    clock = Clock()
    fake = FakeScheduler(clock=clock)
    return World(root, clock, fake, Wire(fake))
