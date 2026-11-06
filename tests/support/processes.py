from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import NamedTuple

SPAWN = multiprocessing.get_context("spawn")


class Gate(NamedTuple):
    ready: Connection
    release: Connection


def gate() -> tuple[Gate, Gate]:
    """Return (worker side, test side) of a ready/release pair."""
    ready_test, ready_worker = SPAWN.Pipe(duplex=False)
    release_worker, release_test = SPAWN.Pipe(duplex=False)
    return Gate(ready_worker, release_worker), Gate(ready_test, release_test)


def spawn(target: Callable[..., None], *args: object) -> BaseProcess:
    process = SPAWN.Process(target=target, args=args)
    process.start()
    return process


def launch(target: Callable[..., None], *args: object) -> tuple[BaseProcess, Connection]:
    """Spawn `target(*args, result)`; return the process and the read end of `result`."""
    result_test, result_worker = SPAWN.Pipe(duplex=False)
    process = spawn(target, *args, result_worker)
    result_worker.close()
    return process, result_test


def race(*racers: Callable[[Connection, Connection], None]) -> list[str]:
    """Spawn each `racer(start, result)`, release all at once, and return results in order."""
    started: list[tuple[BaseProcess, Connection, Connection]] = []
    for racer in racers:
        start_worker, start_test = SPAWN.Pipe(duplex=False)
        process, result = launch(racer, start_worker)
        start_worker.close()
        started.append((process, start_test, result))
    for _, start, _ in started:
        start.send("start")
        start.close()
    outcomes: list[str] = []
    for process, _, result in started:
        assert result.poll(10)
        outcomes.append(result.recv())
        process.join(timeout=10)
        assert process.exitcode == 0
    return outcomes
