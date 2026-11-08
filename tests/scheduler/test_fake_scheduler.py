"""``servatus.testing.FakeScheduler`` answers the exact commands ``Scheduler`` issues."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath

import pytest
from scheduler_fixtures import allocation
from support.builders import resources, target, tasks

from servatus.campaign._config import Target
from servatus.campaign._evidence import Execution, Observation, evidenced_job, observe
from servatus.campaign._remote import Completed
from servatus.campaign._scheduler import Scheduler, find_argv
from servatus.campaign._script import (
    Chained,
    dependency_options,
    log_path,
    render_batch,
    sbatch_argv,
)
from servatus.errors import ConfigurationError, EvidenceConflict, Unavailable
from servatus.testing import FakeJob, FakeScheduler

SBATCH = "/opt/slurm/bin/sbatch"


class Harness:
    def __init__(self) -> None:
        self.fake = FakeScheduler()
        self.target: Target = target()
        self.root = self.target.log_root
        self.scheduler = Scheduler(self.fake(self.target), self.target)

    def request(self, allocation_id: str, count: int = 2) -> tuple[tuple[str, ...], bytes]:
        argv = sbatch_argv(self.target, resources(), count, allocation_id)
        script = render_batch(self.target, resources(), tasks(count), allocation_id)
        return argv, script

    def submit(self, allocation_id: str, count: int = 2) -> int:
        return self.scheduler.submit(*self.request(allocation_id, count))

    def observe(self, allocation_id: str, job: int, count: int = 2) -> Observation:
        rows = self.scheduler.queue([allocation_id])[allocation_id]
        markers = self.scheduler.markers({allocation_id: count})[allocation_id]
        return observe(allocation_id, job, count, rows, markers)

    def summary(self, allocation_id: str, job: int, count: int = 2) -> tuple[object, ...]:
        """The queue state, whether the start and stop markers exist, and each exit code."""
        seen = self.observe(allocation_id, job, count)
        codes = tuple(None if end is None else end.exit_code for end in seen.ends)
        return seen.slurm_state, seen.started_at is not None, seen.stopped_at is not None, codes

    def executions(self, allocation_id: str, job: int, count: int = 2) -> list[object]:
        seen = self.observe(allocation_id, job, count)
        return [seen.execution(slot)[:2] for slot in range(count)]

    def marker(self, allocation_id: str, job: int, suffix: str) -> str:
        return f"{self.root}/{allocation_id}.{job}.{suffix}"

    def markers(self) -> set[str]:
        return {path for path in self.fake.files if path.endswith((".start", ".stop", ".end"))}


def test_fake_is_a_connect_function_returning_a_transport() -> None:
    fake = FakeScheduler(version="slurm 24.11.5")
    transport = fake(target())
    assert transport.run((SBATCH, "--version")) == Completed(0, b"slurm 24.11.5\n", b"")
    with pytest.raises(ConfigurationError, match="connect needs a Target"):
        fake("login.example.edu")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ConfigurationError, match="first_job_id"):
        FakeScheduler(first_job_id=0)


def test_full_lifecycle_through_the_real_scheduler() -> None:
    harness = Harness()
    fake, scheduler, a = harness.fake, harness.scheduler, allocation(1)
    assert scheduler.ping() == "slurm 23.11.4"
    argv, script = harness.request(a)
    accepted, _, stderr = scheduler.test_only(argv, script)
    assert accepted and stderr.startswith("sbatch: Job 1000 to start at")
    assert fake.jobs == ()

    job = scheduler.submit(argv, script)
    assert job == 1000
    assert fake.job(job) == FakeJob(job, f"servatus-{a}", argv, script, 2, "PENDING", True)
    assert fake.jobs == (fake.job(job),)
    assert harness.summary(a, job) == ("PENDING", False, False, (None, None))
    assert harness.executions(a, job) == [(Execution.QUEUED, None)] * 2

    fake.start(job)
    assert harness.markers() == {harness.marker(a, job, "start")}
    assert harness.executions(a, job) == [(Execution.RUNNING, None)] * 2

    fake.finish_step(job, 0, exit_code=3)
    assert harness.marker(a, job, "0.3.end") in fake.files
    assert harness.executions(a, job) == [(Execution.FAILED, 3), (Execution.RUNNING, None)]

    fake.finish(job)
    assert fake.job(job).state == "FAILED"
    assert harness.summary(a, job) == ("FAILED", True, False, (3, 0))
    assert harness.observe(a, job).live

    fake.purge(job)
    assert not fake.job(job).in_queue
    assert scheduler.queue([a]) == {a: ()}
    assert harness.summary(a, job) == (None, True, False, (3, 0))
    assert harness.executions(a, job) == [(Execution.FAILED, 3), (Execution.SUCCEEDED, 0)]

    path = log_path(harness.root, a, job, 1)
    fake.write_file(path, b"\x00log tail\xff")
    assert scheduler.tail(path, 4) == (b"ail\xff", True)
    assert scheduler.tail(path, 64) == (b"\x00log tail\xff", False)
    assert (fake.count("ping"), fake.count("sbatch")) == (1, 2)
    assert fake.calls[0] == (SBATCH, "--version")


def test_finish_settles_completed_and_starts_a_pending_job_first() -> None:
    harness = Harness()
    job = harness.submit(allocation(1))
    harness.fake.finish(job)
    assert harness.fake.job(job).state == "COMPLETED"
    assert harness.markers() == {
        harness.marker(allocation(1), job, suffix) for suffix in ("start", "0.0.end", "1.0.end")
    }
    assert harness.executions(allocation(1), job) == [(Execution.SUCCEEDED, 0)] * 2


def test_interrupting_a_pending_job_writes_no_marker_and_it_never_started() -> None:
    harness = Harness()
    job = harness.submit(allocation(1))
    harness.fake.interrupt(job)
    assert harness.fake.job(job).state == "CANCELLED"
    assert harness.markers() == set()
    assert harness.observe(allocation(1), job).slurm_state == "CANCELLED"
    harness.fake.purge(job)
    assert harness.summary(allocation(1), job) == (None, False, False, (None, None))
    assert harness.executions(allocation(1), job) == [(Execution.STOPPED, None)] * 2


def test_interrupting_a_running_job_writes_its_stop_marker() -> None:
    harness = Harness()
    job = harness.submit(allocation(1))
    harness.fake.start(job)
    harness.fake.finish_step(job, 1)
    harness.fake.interrupt(job, state="TIMEOUT")
    assert harness.fake.job(job).state == "TIMEOUT"
    assert harness.marker(allocation(1), job, "stop") in harness.fake.files
    harness.fake.purge(job)
    assert harness.executions(allocation(1), job) == [
        (Execution.STOPPED, None),
        (Execution.SUCCEEDED, 0),
    ]


def test_a_lost_job_writes_nothing_and_is_lost_once_purged() -> None:
    harness = Harness()
    job = harness.submit(allocation(1))
    harness.fake.start(job)
    harness.fake.lose(job)
    assert harness.fake.job(job).state == "NODE_FAIL"
    assert harness.markers() == {harness.marker(allocation(1), job, "start")}
    assert harness.executions(allocation(1), job) == [(Execution.ENDING, None)] * 2
    harness.fake.purge(job)
    assert harness.executions(allocation(1), job) == [(Execution.LOST, None)] * 2


def test_cancel_interrupts_only_live_jobs_with_the_allocation_name() -> None:
    harness = Harness()
    pending = harness.submit(allocation(1))
    running = harness.submit(allocation(2))
    finished = harness.submit(allocation(3))
    harness.fake.start(running)
    harness.fake.finish(finished)

    harness.scheduler.cancel(allocation(9))
    assert [job.state for job in harness.fake.jobs] == ["PENDING", "RUNNING", "COMPLETED"]

    for index in (1, 2, 3):
        harness.scheduler.cancel(allocation(index))
    assert [job.state for job in harness.fake.jobs] == ["CANCELLED", "CANCELLED", "COMPLETED"]
    assert harness.marker(allocation(1), pending, "stop") not in harness.fake.files
    assert harness.marker(allocation(2), running, "stop") in harness.fake.files
    assert harness.fake.calls[-1] == (
        "/opt/slurm/bin/scancel",
        "--me",
        f"--name=servatus-{allocation(3)}",
    )


def test_cancel_by_name_never_touches_a_reused_job_number() -> None:
    harness = Harness()
    first = harness.submit(allocation(1))
    harness.fake.start(first)
    harness.fake.set_next_job_id(first)
    second = harness.submit(allocation(2))
    assert first == second
    harness.scheduler.cancel(allocation(1))
    assert [job.state for job in harness.fake.jobs] == ["CANCELLED", "PENDING"]


def test_a_second_job_with_an_existing_identity_is_a_conflict() -> None:
    harness = Harness()
    first = harness.submit(allocation(1))
    original = harness.fake.job(first)
    reply = harness.fake.run(original.argv, stdin=original.script)
    assert reply == Completed(0, b"1001\n", b"")
    rows = harness.scheduler.queue([allocation(1)])[allocation(1)]
    assert [row.job for row in rows] == [1000, 1001]
    with pytest.raises(EvidenceConflict, match="Slurm lists jobs 1000, 1001"):
        harness.observe(allocation(1), first)
    with pytest.raises(EvidenceConflict, match="evidence names jobs 1000, 1001"):
        evidenced_job(allocation(1), rows, ())


def test_files_carry_mtimes_from_the_fakes_clock() -> None:
    fake = FakeScheduler()
    start = fake.now()
    fake.write_file("/logs/a")
    fake.advance(minutes=3)
    fake.write_file("/logs/b", b"x")
    fake.write_file("/logs/c", at=start + timedelta(seconds=1, microseconds=5))
    assert fake.mtime("/logs/a") == start == datetime(2030, 1, 1, tzinfo=UTC)
    assert fake.mtime("/logs/b") == start + timedelta(minutes=3)
    listed = fake.run(find_argv(PurePosixPath("/logs"), ("*",))).stdout.decode().splitlines()
    assert listed[2] == "1893456001.0000050000 /logs/c"
    with pytest.raises(ConfigurationError, match="never runs backwards"):
        fake.advance(seconds=-1)
    with pytest.raises(ConfigurationError, match="aware UTC"):
        fake.write_file("/logs/d", at=datetime(2030, 1, 1))  # noqa: DTZ001
    shared = datetime(2031, 1, 1, tzinfo=UTC)
    clocked = FakeScheduler(clock=lambda: shared)
    clocked.write_file("/logs/e")
    assert clocked.mtime("/logs/e") == shared == clocked.now()
    with pytest.raises(ConfigurationError, match="advance the clock"):
        clocked.advance(seconds=1)
    with pytest.raises(ConfigurationError, match="clock must be callable"):
        FakeScheduler(clock=shared)  # pyright: ignore[reportArgumentType]


def test_find_lists_only_marker_files_directly_in_the_root() -> None:
    harness = Harness()
    root = str(harness.root)
    wanted = [f"{root}/{allocation(1)}.5.start", f"{root}/b.5.stop", f"{root}/a.5.0.0.end"]
    for path in [
        *wanted,
        f"{root}/sub/{allocation(1)}.5.stop",
        f"{root}2/{allocation(1)}.5.stop",
        f"{root}/{allocation(1)}-5.out",
        f"{root}/{allocation(1)}.5.start.tmp",
        f"/elsewhere/{allocation(1)}.5.start",
    ]:
        harness.fake.write_file(path)
    every = ("*.start", "*.stop", "*.end")
    stamp = "1893456000.0000000000"
    listing = harness.fake.run(find_argv(harness.root, every)).stdout
    assert listing == "".join(f"{stamp} {path}\n" for path in sorted(wanted)).encode()
    prefixed = harness.fake.run(find_argv(harness.root, ("a*.start", "a*.stop", "a*.end"))).stdout
    assert prefixed == f"{stamp} {root}/a.5.0.0.end\n".encode()
    named = harness.fake.run(find_argv(harness.root, (f"{allocation(1)}.*",))).stdout
    assert (
        named
        == (
            f"{stamp} {root}/{allocation(1)}.5.start\n{stamp} {root}/{allocation(1)}.5.start.tmp\n"
        ).encode()
    )

    harness.fake.remove_file(wanted[0])
    harness.fake.remove_file(f"{root}/{allocation(1)}.5.start.tmp")
    assert wanted[0] not in harness.fake.files
    assert harness.scheduler.markers({allocation(1): 1}) == {allocation(1): ()}


@pytest.mark.parametrize(
    "argv",
    [
        ("/opt/slurm/bin/squeue", "--all"),
        (
            "/opt/slurm/bin/squeue",
            "--me",
            "--all",
            "--states=all",
            "--noheader",
            "--name=servatus-x",
            "--format=%i|%j|%T|%r",
        ),
        ("/opt/slurm/bin/squeue", "--me", "--states=all", "--all", "--noheader", "--name=a", "x"),
        ("/opt/slurm/bin/scancel", "--me", "--name=servatus-x", "42"),
        ("/opt/slurm/bin/scancel", "--name=servatus-x"),
        ("/opt/slurm/bin/scancel", "42"),
        ("/usr/bin/find", "/logs", "-maxdepth", "1", "-type", "f", "(", "-name", "*.start", ")"),
        ("/usr/bin/find", "-H", "/logs", "-maxdepth", "2", "-type", "f", "(", "-name", "x", ")"),
        (*find_argv(PurePosixPath("/logs"), ("*.end",))[:-3], "-print0", ")"),
        find_argv(PurePosixPath("/logs"), ("*.end",))[:-2],
        ("/usr/bin/tail", "-n", "5", "--", "/log"),
    ],
)
def test_argv_drift_is_an_assertion_error(argv: tuple[str, ...]) -> None:
    with pytest.raises(AssertionError, match="does not understand"):
        FakeScheduler().run(argv)


def test_sbatch_drift_is_an_assertion_error() -> None:
    harness = Harness()
    argv, script = harness.request(allocation(1))
    for drifted in (
        (*argv, f"--comment=servatus-{allocation(1)}"),
        tuple(word for word in argv if word != "--no-requeue"),
        tuple(word for word in argv if word != "--parsable"),
        (*argv, "--requeue"),
    ):
        with pytest.raises(AssertionError, match="does not understand"):
            harness.fake.run(drifted, stdin=script)
    with pytest.raises(AssertionError, match="--export=NIL and a /bin/sh script"):
        harness.fake.run(argv, stdin=b"#!/bin/bash\n")
    assert harness.fake.jobs == ()


def test_failed_submission_creates_no_job() -> None:
    harness = Harness()
    harness.fake.fail_next("sbatch")
    with pytest.raises(Unavailable, match="injected failure of sbatch"):
        harness.submit(allocation(1))
    rejected = Completed(1, b"", b"sbatch: error: Batch job submission failed\n")
    harness.fake.fail_next("sbatch", rejected)
    with pytest.raises(Unavailable, match="status 1: sbatch: error: Batch job submission"):
        harness.submit(allocation(1))
    assert harness.fake.jobs == ()
    assert harness.scheduler.queue([allocation(1)]) == {allocation(1): ()}
    assert harness.submit(allocation(1)) == 1000


def test_lost_submission_reply_leaves_the_job_and_its_evidence() -> None:
    harness = Harness()
    harness.fake.lose_next_reply("sbatch")
    with pytest.raises(Unavailable, match="reply from sbatch was lost"):
        harness.submit(allocation(1))
    (accepted,) = harness.fake.jobs
    rows = harness.scheduler.queue([allocation(1)])[allocation(1)]
    assert evidenced_job(allocation(1), rows, ()) == accepted.job == 1000


def test_injected_failures_reach_observation_commands() -> None:
    harness = Harness()
    harness.fake.fail_next("squeue")
    with pytest.raises(Unavailable, match="injected failure of squeue"):
        harness.scheduler.queue([allocation(1)])
    harness.fake.fail_next("find", Completed(1, b"", b"find: /x: No such file or directory\n"))
    with pytest.raises(Unavailable, match="exit status 1: find: /x"):
        harness.scheduler.markers({allocation(1): 1})
    harness.fake.lose_next_reply("scancel")
    with pytest.raises(Unavailable, match="reply from scancel was lost"):
        harness.scheduler.cancel(allocation(1))
    assert harness.scheduler.queue([allocation(1)]) == {allocation(1): ()}


def test_the_ping_has_its_own_control_key() -> None:
    harness = Harness()
    harness.fake.fail_next("sbatch")
    harness.fake.lose_next_reply("sbatch")
    assert harness.scheduler.ping() == "slurm 23.11.4"
    harness.fake.fail_next("ping", Completed(127, b"", b"sbatch: not found\n"))
    with pytest.raises(Unavailable, match="exit status 127"):
        harness.scheduler.ping()
    harness.fake.lose_next_reply("ping")
    with pytest.raises(Unavailable, match="reply from ping was lost"):
        harness.scheduler.ping()
    with pytest.raises(Unavailable, match="injected failure of sbatch"):
        harness.submit(allocation(1))
    assert (harness.fake.count("ping"), harness.fake.count("sbatch")) == (3, 1)


def test_fake_enforces_real_bounds_and_rejects_unknown_commands() -> None:
    fake = FakeScheduler()
    transport = fake(target())
    with pytest.raises(ConfigurationError, match="bound of 32 arguments"):
        transport.run(("/opt/slurm/bin/squeue",) * 33)
    assert fake.calls == ()
    with pytest.raises(AssertionError, match="does not emulate"):
        transport.run(("/opt/slurm/bin/sacct", "--version"))
    with pytest.raises(Unavailable, match="byte bound"):
        transport.run((SBATCH, "--version"), max_stdout=3)


def test_missing_log_is_unavailable() -> None:
    harness = Harness()
    with pytest.raises(Unavailable, match="log is unavailable"):
        harness.scheduler.tail(PurePosixPath("/cluster/logs/none.out"), 16)


def test_controls_reject_impossible_transitions() -> None:
    harness = Harness()
    fake = harness.fake
    with pytest.raises(ConfigurationError, match="no job 7"):
        fake.start(7)
    job = harness.submit(allocation(1))
    with pytest.raises(ConfigurationError, match="no running step in slot 0"):
        fake.finish_step(job, 0)
    fake.start(job)
    with pytest.raises(ConfigurationError, match="is not pending"):
        fake.start(job)
    for slot in (-1, 2):
        with pytest.raises(ConfigurationError, match=f"no running step in slot {slot}"):
            fake.finish_step(job, slot)
    for code in (-1, 256, True):
        with pytest.raises(ConfigurationError, match="exit_code must be an integer from 0 to 255"):
            fake.finish_step(job, 0, code)
    fake.finish_step(job, 0)
    with pytest.raises(ConfigurationError, match="no running step in slot 0"):
        fake.finish_step(job, 0)
    fake.finish(job)
    with pytest.raises(ConfigurationError, match="is not running"):
        fake.finish(job)
    with pytest.raises(ConfigurationError, match="already ended"):
        fake.interrupt(job)
    with pytest.raises(ConfigurationError, match="already ended"):
        fake.lose(job)
    with pytest.raises(ConfigurationError, match="job_id"):
        fake.set_next_job_id(0)


# --- chains ----------------------------------------------------------------------------------

SIGNALLED = resources(gpus=0, time_limit="01:00:00", signal_before_end="00:10:00")


def chain_link(
    harness: Harness, index: int, count: int, after: tuple[int, ...] = (), slots: int = 1
) -> int:
    """Submit link ``index`` of a ``count``-link chain of ``allocation(n)`` identities."""
    chain = Chained(
        index,
        None,
        (count,) * slots,
        (timedelta(seconds=30),) * slots,
        allocation(index + 1) if index < count else None,
    )
    argv = sbatch_argv(harness.target, SIGNALLED, slots, allocation(index), chain)
    script = render_batch(harness.target, SIGNALLED, tasks(slots), allocation(index), chain)
    return harness.scheduler.submit((*argv, *dependency_options(after)), script)


def handoff(harness: Harness, index: int, slot: int = 0) -> str:
    return f"{harness.root}/{allocation(index)}.{slot}.go"


def test_a_chain_link_waits_for_its_dependencies_and_slurm_prunes_it() -> None:
    harness = Harness()
    first = chain_link(harness, 1, 3)
    second = chain_link(harness, 2, 3, (first,))
    third = chain_link(harness, 3, 3, (first, second))
    assert [job.dependencies for job in harness.fake.jobs] == [(), (first,), (first, second)]
    with pytest.raises(ConfigurationError, match=f"job {second} waits for its dependencies"):
        harness.fake.start(second)
    harness.fake.start(first)
    harness.fake.advance(minutes=50)
    harness.fake.signal_limit(first)
    assert harness.marker(allocation(1), first, "limit") in harness.fake.files
    harness.fake.advance(minutes=10)
    harness.fake.interrupt(first, "TIMEOUT")
    assert harness.fake.files[handoff(harness, 2)] == b"3630\n"
    harness.fake.start(second)
    harness.fake.advance(minutes=1)
    harness.fake.finish(second)
    assert [job.state for job in harness.fake.jobs] == ["TIMEOUT", "COMPLETED", "CANCELLED"]
    assert handoff(harness, 3) not in harness.fake.files
    assert harness.marker(allocation(3), third, "start") not in harness.fake.files


def test_a_link_runs_only_handed_slots_and_exits_0_with_none() -> None:
    harness = Harness()
    first = chain_link(harness, 1, 3, slots=2)
    second = chain_link(harness, 2, 3, (first,), slots=2)
    third = chain_link(harness, 3, 3, (first, second), slots=2)
    harness.fake.start(first)
    harness.fake.finish_step(first, 0, 75)
    harness.fake.finish_step(first, 1, 0)
    harness.fake.finish(first)
    assert harness.fake.job(first).state == "FAILED"
    harness.fake.write_file(handoff(harness, 2, 1), b"012\n")
    harness.fake.write_file(handoff(harness, 2), b" 7\t\nignored")
    harness.fake.start(second)
    with pytest.raises(ConfigurationError, match="no running step in slot 1"):
        harness.fake.finish_step(second, 1)
    harness.fake.finish(second, exit_code=1)
    assert harness.fake.job(second).state == "COMPLETED"
    assert harness.fake.job(third).state == "CANCELLED"


def test_a_link_submitted_after_an_earlier_link_exited_0_is_deleted() -> None:
    harness = Harness()
    first = chain_link(harness, 1, 2)
    harness.fake.finish(first)
    second = chain_link(harness, 2, 2, (first,))
    assert harness.fake.job(second).state == "CANCELLED"


def test_a_link_with_nothing_handed_completes_as_it_starts() -> None:
    harness = Harness()
    first = chain_link(harness, 1, 2)
    second = chain_link(harness, 2, 2, (first,))
    harness.fake.start(first)
    harness.fake.interrupt(first)
    harness.fake.start(second)
    assert harness.fake.job(second).state == "COMPLETED"
    harness.fake.finish(second)
    assert harness.fake.job(second).state == "COMPLETED"


def test_the_pre_limit_signal_reaches_a_plain_job_without_a_limit_marker() -> None:
    harness = Harness()
    job = harness.submit(allocation(1))
    with pytest.raises(ConfigurationError, match="is not running"):
        harness.fake.signal_limit(job)
    harness.fake.start(job)
    harness.fake.signal_limit(job)
    harness.fake.interrupt(job, "TIMEOUT")
    assert harness.markers() == {harness.marker(allocation(1), job, s) for s in ("start", "stop")}
    assert not any(path.endswith(".go") for path in harness.fake.files)
    with pytest.raises(ConfigurationError, match="is not running"):
        harness.fake.finish(job)


def test_a_dependency_on_a_job_slurm_forgot_is_refused() -> None:
    harness = Harness()
    first = chain_link(harness, 1, 2)
    harness.fake.finish(first)
    harness.fake.purge(first)
    with pytest.raises(Unavailable, match="Job dependency problem"):
        chain_link(harness, 2, 2, (first,))
    with pytest.raises(Unavailable, match="Job dependency problem"):
        chain_link(harness, 2, 2, (77,))
    assert len(harness.fake.jobs) == 1


@pytest.mark.parametrize(
    "options",
    [
        ("--dependency=afterok:1000", "--kill-on-invalid-dep=yes"),
        ("--dependency=afternotok:", "--kill-on-invalid-dep=yes"),
        ("--dependency=afternotok:1000", "--kill-on-invalid-dep=no"),
    ],
)
def test_dependency_drift_is_an_assertion_error(options: tuple[str, ...]) -> None:
    harness = Harness()
    harness.submit(allocation(1))
    argv, script = harness.request(allocation(2))
    with pytest.raises(AssertionError, match="does not understand"):
        harness.fake.run((*argv, *options), stdin=script)
