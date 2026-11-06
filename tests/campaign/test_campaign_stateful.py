"""Stateful property: random operator, scheduler, and network histories through the public API.

Checked after every step:

1. durable state always reloads (the strict decoder accepts everything the store wrote);
2. history is never erased: earlier Attempts keep their place, and a recorded outcome never
   changes;
3. ``quiescent`` holds exactly when no acceptance is unresolved and Slurm lists no job.

Checked at every submission: Servatus never submits a Task while Slurm still lists a job for it.
Operations fail only with the documented recoverable errors.
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from pathlib import Path

from campaign_world import World, cpu_profile, jobs, make_world
from hypothesis import HealthCheck, settings, stateful
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from servatus.campaign import (
    AcceptanceState,
    AttemptStatus,
    Campaign,
    Completed,
    Execution,
    PlannedAllocation,
    Retry,
)
from servatus.errors import PlanRefused, SubmissionInterrupted, Unavailable
from servatus.testing import FakeJob

LOGS = "/cluster/logs/project"
_ENDINGS = ("lose", "fail", "interrupt", "finish")
_MOVES = {
    "PENDING": ("start", *_ENDINGS, "wait"),
    "RUNNING": ("step", *_ENDINGS, "wait"),
}


def allocation_of(job: FakeJob) -> str:
    return job.name.removeprefix("servatus-")


class CampaignMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.root = Path(tempfile.mkdtemp())
        self.world: World = make_world(self.root)
        self.campaign: Campaign = self.world.create(jobs(3), appendable=True)
        self.count = 3
        self.planned: dict[str, PlannedAllocation] = {}
        self.history: tuple[AttemptStatus, ...] = ()

    def teardown(self) -> None:
        shutil.rmtree(self.root)

    def attempts(self) -> tuple[AttemptStatus, ...]:
        return self.campaign.status(scheduler=False).attempts

    def owners(self) -> dict[str, tuple[str, ...]]:
        return {attempt.allocation_id: attempt.task_keys for attempt in self.attempts()}

    def listed_tasks(self) -> set[str]:
        owners = self.owners()
        return {
            key
            for job in self.world.fake.jobs
            if job.in_queue
            for key in owners.get(allocation_of(job), ())
        }

    # --- operator actions ---------------------------------------------------------------------

    @rule(cap=st.integers(1, 3), retry_failed=st.booleans(), retry_lost=st.booleans())
    def plan_and_submit(self, cap: int, retry_failed: bool, retry_lost: bool) -> None:
        status = self.campaign.status()
        lost = [task.key for task in status.tasks if task.execution is Execution.LOST]
        retry = lost if retry_lost and lost else Retry.FAILED if retry_failed else ()
        try:
            plan = self.campaign.plan(cpu_profile(), retry=retry, tasks_per_allocation=cap)
        except PlanRefused as error:
            assert "Slurm still lists earlier work" in str(error)
            return
        self.planned.update((item.allocation_id, item) for item in plan.allocations)
        listed = self.listed_tasks()
        before = len(self.world.fake.jobs)
        with contextlib.suppress(SubmissionInterrupted):
            self.campaign.submit(plan)
        owners = self.owners()
        for job in self.world.fake.jobs[before:]:
            duplicated = listed.intersection(owners[allocation_of(job)])
            assert not duplicated, f"job {job.job} duplicates listed work of {sorted(duplicated)}"

    @rule()
    def append(self) -> None:
        if not self.campaign.status(scheduler=False).sealed and self.count < 6:
            self.count += 1
            self.campaign.append(jobs(self.count)[-1:])

    @rule()
    def seal(self) -> None:
        self.campaign.seal()

    @rule(minutes=st.sampled_from([0, 5, 15]))
    def reconcile(self, minutes: int) -> None:
        self.world.clock.advance(minutes=minutes)
        result = self.campaign.reconcile()
        if minutes == 15:
            assert result.complete

    @rule(data=st.data())
    def cancel_a_task(self, data: st.DataObject) -> None:
        submitted = sorted(
            {
                key
                for attempt in self.attempts()
                if attempt.acceptance is not AcceptanceState.UNRESOLVED
                for key in attempt.task_keys
            }
        )
        if submitted:
            self.campaign.cancel(tasks=[data.draw(st.sampled_from(submitted))])

    # --- scheduler and network events ---------------------------------------------------------

    @rule(kind=st.sampled_from(["lost-reply", "rejected", "transport"]))
    def next_submission_misbehaves(self, kind: str) -> None:
        fake = self.world.fake

        def misbehave(_argv: tuple[str, ...]) -> Completed | None:
            if kind == "lost-reply":
                fake.lose_next_reply("sbatch")
                return None
            if kind == "rejected":
                return Completed(1, b"", b"sbatch: error: rejected\n")
            raise Unavailable("connection reset by peer")

        self.world.wire.on("sbatch", misbehave)

    @rule(data=st.data())
    def a_failed_sbatch_arrives_late(self, data: st.DataObject) -> None:
        named = {allocation_of(job) for job in self.world.fake.jobs}
        candidates = [
            attempt.allocation_id
            for attempt in self.attempts()
            if attempt.acceptance is not AcceptanceState.ACCEPTED
            and attempt.allocation_id not in named
        ]
        if candidates:
            self.world.resubmit(self.planned[data.draw(st.sampled_from(candidates))])

    @precondition(lambda self: any(job.in_queue for job in self.world.fake.jobs))
    @rule(data=st.data())
    def slurm_moves_every_listed_job(self, data: st.DataObject) -> None:
        fake = self.world.fake
        for job in [job for job in fake.jobs if job.in_queue]:
            moves = _MOVES.get(job.state, ("purge", "wait"))
            move = data.draw(st.sampled_from(moves), label=f"job {job.job}")
            if move == "start":
                fake.start(job.job)
            elif move == "step":
                prefix = f"{LOGS}/{allocation_of(job)}.{job.job}."
                open_slots = [
                    slot
                    for slot in range(job.task_count)
                    if not any(path.startswith(f"{prefix}{slot}.") for path in fake.files)
                ]
                if open_slots:
                    slot = data.draw(st.sampled_from(open_slots))
                    fake.finish_step(job.job, slot, exit_code=data.draw(st.sampled_from([0, 1])))
            elif move in ("finish", "fail"):
                fake.finish(job.job, exit_code=0 if move == "finish" else 1)
            elif move == "interrupt":
                fake.interrupt(job.job)
            elif move == "lose":
                fake.lose(job.job)
            elif move == "purge":
                fake.purge(job.job)

    # --- invariants ---------------------------------------------------------------------------

    @invariant()
    def durable_state_reloads(self) -> None:
        reopened = Campaign.open(self.world.path).status(scheduler=False)
        assert reopened.revision == self.campaign.status(scheduler=False).revision

    @invariant()
    def history_is_never_erased(self) -> None:
        current = self.attempts()
        assert len(current) >= len(self.history)
        for before, now in zip(self.history, current, strict=False):
            if before.acceptance is AcceptanceState.UNRESOLVED:
                assert (now.allocation_id, now.task_keys, now.intent_at) == (
                    before.allocation_id,
                    before.task_keys,
                    before.intent_at,
                )
            else:
                assert now == before
        self.history = current

    @invariant()
    def quiescence_matches_the_cluster(self) -> None:
        status = self.campaign.status()
        unresolved = any(
            attempt.acceptance is AcceptanceState.UNRESOLVED for attempt in status.attempts
        )
        listed = any(job.in_queue for job in self.world.fake.jobs)
        assert status.quiescent == (not unresolved and not listed)


def test_random_histories_preserve_the_campaign_invariants() -> None:
    stateful.run_state_machine_as_test(  # pyright: ignore[reportUnknownMemberType]
        CampaignMachine,
        settings=settings(
            max_examples=60,
            stateful_step_count=30,
            deadline=None,
            suppress_health_check=[HealthCheck.too_slow],
        ),
    )
