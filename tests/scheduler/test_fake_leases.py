"""``FakeScheduler`` serves leases as their batch script and Slurm would, driven through its own
Transport with the exact sbatch request, batch script, orders, and claim programs Servatus sends."""

from __future__ import annotations

from datetime import timedelta
from pathlib import PurePosixPath

import pytest
from scheduler_fixtures import LOG_ROOT, allocation, listing
from support.builders import resources, target, tasks

from servatus.campaign._remote import Completed
from servatus.campaign._scheduler import ORDER, WITHDRAW, find_argv
from servatus.campaign._script import (
    Chained,
    order_directory,
    render_lease,
    render_order,
    sbatch_argv,
)
from servatus.errors import ConfigurationError, Unavailable
from servatus.testing import FakeScheduler

LEASE = allocation(100)
LIMITS = resources(gpus=0, time_limit="01:00:00", signal_before_end="00:05:00")
CHAIN = Chained(1, None, (1,), (timedelta(0),))


class Lease:
    """One lease of two slots submitted to a fresh fake."""

    def __init__(self) -> None:
        self.fake = FakeScheduler()
        self.target = target()
        self.orders = str(order_directory(self.target.log_root, LEASE))
        argv = sbatch_argv(self.target, LIMITS, 2, LEASE, relay=True)
        self.job = int(self.fake.run(argv, stdin=render_lease(self.target, LIMITS, LEASE)).stdout)

    def order(self, *steps: str) -> None:
        for step in steps:
            script = render_order(self.target, LIMITS, tasks(1)[0], step, CHAIN)
            assert self.fake.run((*ORDER, self.orders, step), stdin=script) == Completed(
                0, b"", b""
            )

    def withdraw(self, step: str, *, stop: bool) -> bytes:
        marker = self.marker(step, "")
        argv = (*WITHDRAW, self.orders, step, f"{marker}void", f"{marker}stop" if stop else "")
        return self.fake.run(argv).stdout

    def running(self, *steps: str) -> tuple[str, ...]:
        """Start the lease with ``steps`` ordered; the steps it took."""
        self.order(*steps)
        self.fake.start(self.job)
        return self.fake.poll(self.job)

    def marker(self, name: str, suffix: str) -> str:
        return f"{LOG_ROOT}/{name}.{self.job}.{suffix}"

    def claim(self, step: str) -> bytes | None:
        return self.fake.files.get(f"{self.orders}/{step}.claim")

    def step_markers(self) -> set[str]:
        return {
            path
            for path in self.fake.files
            if path.startswith(f"{LOG_ROOT}/") and not path.startswith(f"{LOG_ROOT}/{LEASE}")
        }


def test_a_lease_starts_without_taking_orders_and_never_finishes() -> None:
    lease = Lease()
    lease.order(allocation(1))
    assert lease.fake.files[f"{lease.orders}/{allocation(1)}.run"].startswith(b"previous=0\n")
    with pytest.raises(ConfigurationError, match="is not a running lease"):
        lease.fake.poll(lease.job)
    lease.fake.start(lease.job)
    assert lease.fake.job(lease.job).state == "RUNNING"
    assert lease.marker(LEASE, "start") in lease.fake.files
    assert lease.claim(allocation(1)) is None and lease.step_markers() == set()
    with pytest.raises(ConfigurationError, match="ends only when Slurm interrupts it"):
        lease.fake.finish(lease.job)
    assert lease.fake.poll(lease.job) == (allocation(1),)


def test_poll_claims_new_orders_in_name_order_exactly_once() -> None:
    lease = Lease()
    assert lease.running(allocation(3), allocation(2)) == (allocation(2), allocation(3))
    assert lease.claim(allocation(2)) == lease.claim(allocation(3)) == b"taken"
    assert lease.fake.poll(lease.job) == ()
    lease.order(allocation(4), allocation(1))
    assert lease.withdraw(allocation(1), stop=False) == b"withdrawn\n"
    assert lease.fake.poll(lease.job) == (allocation(4),)
    assert lease.claim(allocation(1)) == b"withdrawn"
    assert lease.step_markers() == {lease.marker(allocation(1), "void")}


def test_poll_takes_nothing_after_the_pre_limit_signal() -> None:
    lease = Lease()
    lease.running()
    lease.fake.signal_limit(lease.job)
    assert lease.marker(LEASE, "limit") in lease.fake.files
    lease.order(allocation(1))
    assert lease.fake.poll(lease.job) == ()
    assert lease.claim(allocation(1)) is None


@pytest.mark.parametrize("ending", ["interrupt", "lose"])
def test_poll_refuses_a_lease_that_is_not_running(ending: str) -> None:
    lease = Lease()
    lease.running()
    getattr(lease.fake, ending)(lease.job)
    with pytest.raises(ConfigurationError, match="is not a running lease"):
        lease.fake.poll(lease.job)


def test_poll_refuses_a_job_that_is_not_a_lease() -> None:
    fake = FakeScheduler()
    argv = sbatch_argv(target(), LIMITS, 1, allocation(1))
    job = int(fake.run(argv, stdin=b"#!/bin/sh\n").stdout)
    fake.start(job)
    with pytest.raises(ConfigurationError, match="is not a running lease"):
        fake.poll(job)


def test_steps_write_their_start_and_end_markers_under_the_leases_job() -> None:
    lease = Lease()
    first, second = allocation(1), allocation(2)
    lease.running(first, second)
    lease.fake.begin_step(first)
    assert lease.step_markers() == {lease.marker(first, "start")}
    with pytest.raises(ConfigurationError, match=f"step {first} already began"):
        lease.fake.begin_step(first)
    with pytest.raises(ConfigurationError, match="exit_code must be an integer"):
        lease.fake.end_step(second, 256)
    lease.fake.end_step(first, 3)
    lease.fake.end_step(second)
    assert lease.step_markers() == {
        lease.marker(first, "start"),
        lease.marker(first, "0.3.end"),
        lease.marker(second, "0.0.end"),
    }
    for control in (lease.fake.begin_step, lease.fake.end_step):
        for step in (first, allocation(9)):
            with pytest.raises(ConfigurationError, match=f"no running lease runs step {step}"):
                control(step)


def test_honor_stops_ends_exactly_the_taken_unended_steps_asked_to_stop() -> None:
    lease = Lease()
    running, waiting, quiet, ended, withdrawn = (allocation(index) for index in range(1, 6))
    lease.order(withdrawn)
    assert lease.withdraw(withdrawn, stop=True) == b"withdrawn\n"
    lease.running(running, waiting, quiet, ended)
    lease.fake.begin_step(running)
    lease.fake.begin_step(quiet)
    lease.fake.end_step(ended)
    for step in (running, waiting, ended):
        assert lease.withdraw(step, stop=True) == b"taken\n"
    assert lease.fake.honor_stops(lease.job, exit_code=137) == (running, waiting)
    ends = {path for path in lease.step_markers() if path.endswith(".end")}
    assert ends == {
        lease.marker(running, "0.137.end"),
        lease.marker(waiting, "0.137.end"),
        lease.marker(ended, "0.0.end"),
    }
    assert lease.fake.honor_stops(lease.job) == ()


def test_withdraw_matches_the_real_program() -> None:
    lease = Lease()
    lease.order(allocation(1), allocation(2))
    assert lease.withdraw(allocation(1), stop=True) == b"withdrawn\n"
    stop, void = lease.marker(allocation(1), "stop"), lease.marker(allocation(1), "void")
    first = lease.fake.mtime(stop)
    lease.fake.advance(minutes=1)
    assert lease.withdraw(allocation(1), stop=True) == b"withdrawn\n"
    assert lease.fake.mtime(stop) == first
    assert lease.fake.mtime(void) == first + timedelta(minutes=1)
    lease.fake.start(lease.job)
    assert lease.fake.poll(lease.job) == (allocation(2),)
    assert lease.withdraw(allocation(2), stop=False) == b"taken\n"
    assert lease.claim(allocation(2)) == b"taken"
    assert lease.step_markers() == {stop, void}
    assert lease.withdraw(allocation(2), stop=True) == b"taken\n"
    assert lease.marker(allocation(2), "stop") in lease.fake.files
    assert lease.marker(allocation(2), "void") not in lease.fake.files


def test_interrupt_writes_the_lease_stop_marker_and_no_step_end() -> None:
    lease = Lease()
    lease.running(allocation(1), allocation(2))
    lease.fake.begin_step(allocation(1))
    lease.fake.interrupt(lease.job, "TIMEOUT")
    assert lease.fake.job(lease.job).state == "TIMEOUT"
    assert lease.marker(LEASE, "stop") in lease.fake.files
    assert lease.step_markers() == {lease.marker(allocation(1), "start")}
    with pytest.raises(ConfigurationError, match="no running lease runs step"):
        lease.fake.end_step(allocation(2))


def test_markers_are_listed_with_mtimes_from_the_fake_clock() -> None:
    lease = Lease()
    step = allocation(1)
    lease.running(step)
    lease.fake.begin_step(step)
    lease.fake.advance(minutes=1)
    lease.fake.end_step(step, 75)
    reply = lease.fake.run(find_argv(PurePosixPath(LOG_ROOT), (f"{step}.*", f"{LEASE}.*")))
    ended = listing(f"{step}.{lease.job}.0.75.end", mtime="1893456060.0000000000")
    started = listing(f"{step}.{lease.job}.start", f"{LEASE}.{lease.job}.start")
    assert reply.stdout == ended + started


def test_an_injected_failure_fails_the_next_order_without_writing_it() -> None:
    lease = Lease()
    lease.fake.fail_next("sh")
    with pytest.raises(Unavailable, match="injected failure of sh"):
        lease.order(allocation(1))
    assert f"{lease.orders}/{allocation(1)}.run" not in lease.fake.files
    assert lease.running(allocation(1)) == (allocation(1),)
