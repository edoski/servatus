"""``servatus.testing.FakeScheduler`` answers the exact commands ``Scheduler`` issues."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath

import pytest
from support.builders import resources, target, tasks

from servatus.campaign._config import Target
from servatus.campaign._evidence import AllocationState, JobRef, StepEvidence
from servatus.campaign._remote import Completed
from servatus.campaign._scheduler import AttemptQuery, Scheduler
from servatus.campaign._script import log_path, render_batch, sbatch_argv
from servatus.errors import (
    ConfigurationError,
    EvidenceConflict,
    ReconciliationError,
    Unavailable,
)
from servatus.testing import FakeJob, FakeScheduler

INTENT = datetime(2030, 1, 1, 12, 0, 0, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = INTENT

    def __call__(self) -> datetime:
        return self.now

    def advance(self, minutes: int) -> None:
        self.now += timedelta(minutes=minutes)


def allocation(index: int) -> str:
    return f"{index:024x}"


class Harness:
    def __init__(self, *, cluster: str | None = None) -> None:
        self.clock = Clock()
        self.fake = FakeScheduler(cluster=cluster, clock=self.clock)
        self.target: Target = target()
        self.scheduler = Scheduler(self.fake(self.target), self.target.slurm_bin)

    def request(self, allocation_id: str, count: int = 2) -> tuple[tuple[str, ...], bytes]:
        argv = sbatch_argv(self.target, resources(), count, allocation_id)
        script = render_batch(self.target, resources(), tasks(count), allocation_id)
        return argv, script

    def submit(self, allocation_id: str, count: int = 2) -> JobRef:
        return self.scheduler.submit(*self.request(allocation_id, count))

    def observe(self, allocation_id: str, job: JobRef, count: int = 2) -> tuple[object, ...]:
        query = AttemptQuery(allocation_id, job, INTENT, count)
        observed = self.scheduler.observe([query])[allocation_id]
        evidence = observed.allocation
        return (evidence.state, evidence.retained, evidence.problem, observed.steps)


def test_fake_is_a_connect_function_returning_a_transport() -> None:
    fake = FakeScheduler()
    transport = fake(target())
    assert transport.run(("/opt/slurm/bin/sbatch", "--version")) == Completed(
        0, b"slurm 23.11.4\n", b""
    )
    with pytest.raises(ConfigurationError, match="connect needs a Target"):
        fake("login.example.edu")  # pyright: ignore[reportArgumentType]


def test_full_lifecycle_through_the_real_scheduler() -> None:
    harness = Harness()
    fake, scheduler = harness.fake, harness.scheduler
    assert scheduler.ping() == "slurm 23.11.4"
    argv, script = harness.request(allocation(1))
    accepted, _, stderr = scheduler.test_only(argv, script)
    assert accepted and stderr.startswith("sbatch: Job 1000 to start at")
    assert fake.jobs == ()

    job = scheduler.submit(argv, script)
    assert job == JobRef(1000)
    assert fake.job(job) == FakeJob(job, f"servatus-{allocation(1)}", argv, script, 2, "PENDING", 1)
    assert harness.observe(allocation(1), job) == (AllocationState.QUEUED, True, None, (None, None))

    harness.clock.advance(1)
    fake.start(job)
    running = StepEvidence(AllocationState.RUNNING, "RUNNING", "0:0")
    assert harness.observe(allocation(1), job) == (
        AllocationState.RUNNING,
        True,
        None,
        (running, running),
    )

    # Regression (wedge): a requeue inside the submission window stays observable.
    harness.clock.advance(10)
    fake.requeue(job)
    assert fake.job(job).incarnations == 2
    assert harness.observe(allocation(1), job)[:3] == (AllocationState.QUEUED, True, None)

    harness.clock.advance(5)
    fake.start(job)
    fake.finish_step(job, 0, "FAILED", exit_code="1:0")
    harness.clock.advance(5)
    fake.finish(job)
    assert harness.observe(allocation(1), job) == (
        AllocationState.SUCCEEDED,
        False,
        None,
        (
            StepEvidence(AllocationState.FAILED, "FAILED", "1:0"),
            StepEvidence(AllocationState.SUCCEEDED, "COMPLETED", "0:0"),
        ),
    )

    assert scheduler.identify(allocation(1), INTENT) == job
    path = log_path(harness.target.log_root, allocation(1), job.job_id, 1)
    fake.write_log(path, b"\x00log tail\xff")
    assert scheduler.tail(path, 4) == (b"ail\xff", True)
    assert scheduler.tail(path, 64) == (b"\x00log tail\xff", False)
    scheduler.cancel(allocation(1), job)  # already finished: nothing to cancel
    assert fake.job(job).state == "COMPLETED"
    assert fake.count("sbatch") == 3
    assert fake.calls[0] == ("/opt/slurm/bin/sbatch", "--version")


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
    with pytest.raises(ReconciliationError, match="0 jobs"):
        harness.scheduler.identify(allocation(1), INTENT)


def test_lost_submission_reply_is_recoverable_by_identity() -> None:
    harness = Harness(cluster="alpha")
    harness.fake.lose_next_reply("sbatch")
    with pytest.raises(Unavailable, match="reply from sbatch was lost"):
        harness.submit(allocation(1))
    (accepted,) = harness.fake.jobs
    assert (
        harness.scheduler.identify(allocation(1), INTENT) == accepted.job == JobRef(1000, "alpha")
    )


def test_federated_cluster_routes_every_query() -> None:
    harness = Harness(cluster="alpha")
    job = harness.submit(allocation(1))
    assert job == JobRef(1000, "alpha")
    harness.fake.start(job)
    assert harness.observe(allocation(1), job)[0] is AllocationState.RUNNING
    assert all(
        "--clusters=alpha" in call
        for call in harness.fake.calls
        if call[0].endswith(("squeue", "sacct"))
    )
    harness.fake.finish(job, "TIMEOUT", exit_code="0:1")
    assert harness.observe(allocation(1), job)[:2] == (AllocationState.FAILED, False)
    foreign = AttemptQuery(allocation(1), JobRef(1000, "beta"), INTENT, 2)
    assert harness.scheduler.observe([foreign])[allocation(1)].allocation.state is (
        AllocationState.UNKNOWN
    )


def test_cancel_is_bound_to_the_allocation_name() -> None:
    harness = Harness()
    job = harness.submit(allocation(1))
    harness.fake.start(job)
    harness.scheduler.cancel(allocation(2), job)  # a different allocation's name: no effect
    assert harness.fake.job(job).state == "RUNNING"
    harness.scheduler.cancel(allocation(1), job)
    assert harness.fake.job(job).state == "CANCELLED by 1000"
    state, retained, problem, steps = harness.observe(allocation(1), job)
    assert (state, retained, problem) == (AllocationState.CANCELLED, False, None)
    cancelled = StepEvidence(AllocationState.CANCELLED, "CANCELLED by 1000", "0:15")
    assert steps == (cancelled, cancelled)


def test_forgotten_job_is_unknown_and_not_retained() -> None:
    harness = Harness()
    job = harness.submit(allocation(1))
    harness.fake.forget(job)
    assert harness.observe(allocation(1), job) == (
        AllocationState.UNKNOWN,
        False,
        None,
        (None, None),
    )
    harness.scheduler.cancel(allocation(1), job)  # unknown to Slurm: nothing to cancel


def test_accounting_lag_prefers_the_finished_queue_state_and_stays_retained() -> None:
    # Regression (lag): squeue reported COMPLETED while slurmdbd still said RUNNING, which used to
    # abort observation as a conflict.
    harness = Harness()
    job = harness.submit(allocation(1))
    harness.fake.start(job)
    harness.fake.finish(job, in_queue=True, accounted=False)
    assert harness.observe(allocation(1), job)[:3] == (AllocationState.SUCCEEDED, True, None)
    harness.fake.finish(job)
    assert harness.observe(allocation(1), job)[:3] == (AllocationState.SUCCEEDED, False, None)


def test_started_job_without_accounting_is_still_retained() -> None:
    harness = Harness()
    job = harness.submit(allocation(1))
    harness.fake.start(job, accounted=False)
    assert harness.observe(allocation(1), job)[:3] == (AllocationState.RUNNING, True, None)


def test_reused_job_numbers_stay_bound_to_their_allocations() -> None:
    harness = Harness()
    first = harness.submit(allocation(1))
    harness.fake.finish(first, "FAILED", exit_code="1:0")
    harness.fake.set_next_job_id(first.job_id)
    second = harness.submit(allocation(2))
    harness.fake.finish(second)
    assert first == second
    queries = [AttemptQuery(allocation(index), first, INTENT, 2) for index in (1, 2)]
    observed = harness.scheduler.observe(queries)
    assert observed[allocation(1)].allocation.state is AllocationState.FAILED
    assert observed[allocation(2)].allocation.state is AllocationState.SUCCEEDED
    assert harness.fake.count("squeue") == 2  # one batch per reuse of the number


def test_live_foreign_job_under_a_queried_number_is_an_identity_violation() -> None:
    harness = Harness()
    first = harness.submit(allocation(1))
    harness.fake.finish(first)
    harness.fake.set_next_job_id(first.job_id)
    harness.submit(allocation(2))
    with pytest.raises(EvidenceConflict, match="unrelated allocation"):
        harness.observe(allocation(1), first)


def test_batches_are_counted_per_command() -> None:
    harness = Harness()
    queries = [
        AttemptQuery(allocation(index), harness.submit(allocation(index), 1), INTENT, 1)
        for index in range(20)
    ]
    observed = harness.scheduler.observe(queries)
    assert len(observed) == 20
    assert (harness.fake.count("squeue"), harness.fake.count("sacct")) == (2, 4)


def test_fake_enforces_real_bounds_and_rejects_unknown_commands() -> None:
    fake = FakeScheduler()
    transport = fake(target())
    with pytest.raises(ConfigurationError, match="bound of 32 arguments"):
        transport.run(("/opt/slurm/bin/squeue",) * 33)
    assert fake.calls == ()
    with pytest.raises(AssertionError, match="does not emulate"):
        transport.run(("/opt/slurm/bin/srun", "--version"))
    with pytest.raises(AssertionError, match="does not understand"):
        transport.run(("/opt/slurm/bin/squeue", "--all"))
    with pytest.raises(Unavailable, match="byte bound"):
        transport.run(("/opt/slurm/bin/sbatch", "--version"), max_stdout=3)


def test_missing_log_is_unavailable() -> None:
    harness = Harness()
    with pytest.raises(Unavailable, match="log is unavailable"):
        harness.scheduler.tail(PurePosixPath("/cluster/logs/none.out"), 16)


def test_controls_reject_unknown_jobs_and_unstarted_steps() -> None:
    harness = Harness()
    with pytest.raises(ConfigurationError, match="no job 7"):
        harness.fake.start(7)
    job = harness.submit(allocation(1))
    with pytest.raises(ConfigurationError, match="no started step for slot 0"):
        harness.fake.finish_step(job, 0)
    with pytest.raises(ConfigurationError, match="job_id"):
        harness.fake.set_next_job_id(0)
