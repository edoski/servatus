"""Stateful property: random operator, scheduler, and network histories through the public API.

Invariants checked after every step:

1. durable state always reloads (the strict decoder accepts everything the store wrote);
2. at most one unfinished Slurm job per Task, unless the operator acknowledged duplicate risk;
3. operations fail only with the documented recoverable errors.
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from pathlib import Path

from campaign_world import World, cpu_profile, jobs, make_world
from hypothesis import HealthCheck, settings, stateful
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
)

from servatus.campaign import AcceptanceState, AllocationState, Campaign, Completed
from servatus.errors import (
    PlanRefused,
    ReconciliationError,
    StalePlan,
    SubmissionInterrupted,
    Unavailable,
)
from servatus.testing import FakeJob

_TERMINAL = frozenset({"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"})
_ENDINGS = ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED")


def unfinished(job: FakeJob) -> bool:
    return job.state.split(" ", 1)[0] not in _TERMINAL


class CampaignMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.root = Path(tempfile.mkdtemp())
        self.world: World = make_world(self.root)
        self.campaign: Campaign = self.world.create(jobs(3), appendable=True)
        self.count = 3
        self.acknowledged: set[str] = set()

    def teardown(self) -> None:
        shutil.rmtree(self.root)

    # --- operator actions ---------------------------------------------------------------------

    @rule(cap=st.integers(1, 3), retry_failed=st.booleans(), acknowledge=st.booleans())
    def plan_and_submit(self, cap: int, retry_failed: bool, acknowledge: bool) -> None:
        status = self.campaign.status()
        failed = {AllocationState.FAILED, AllocationState.CANCELLED}
        retry = [task.key for task in status.tasks if retry_failed and task.execution in failed]
        unknown = [
            task.key
            for task in status.tasks
            if acknowledge and task.execution is AllocationState.UNKNOWN
        ]
        try:
            plan = self.campaign.plan(
                cpu_profile(),
                retry=retry + unknown,
                allow_duplicate_risk=unknown,
                tasks_per_allocation=cap,
            )
        except PlanRefused:
            return
        self.acknowledged.update(plan.duplicate_risk)
        with contextlib.suppress(SubmissionInterrupted, Unavailable, StalePlan):
            self.campaign.submit(plan)

    @rule()
    def append(self) -> None:
        if not self.campaign.status(scheduler=False).sealed and self.count < 6:
            self.count += 1
            self.campaign.append(jobs(self.count)[-1:])

    @rule()
    def seal(self) -> None:
        self.campaign.seal()

    @rule()
    def resolve_unresolved_truthfully(self) -> None:
        names = {job.name for job in self.world.fake.jobs}
        for attempt in self.campaign.status(scheduler=False).attempts:
            if attempt.acceptance is not AcceptanceState.UNRESOLVED:
                continue
            if f"servatus-{attempt.allocation_id}" in names:
                # A forgotten job cannot be identified; its allocation stays unresolved.
                with contextlib.suppress(ReconciliationError):
                    self.campaign.reconcile(attempt.allocation_id)
            else:
                self.campaign.mark_not_submitted(attempt.allocation_id)

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

    @precondition(lambda self: any(unfinished(job) for job in self.world.fake.jobs))
    @rule(
        data=st.data(),
        event=st.sampled_from(["START", "FORGET", "REQUEUE", *_ENDINGS]),
    )
    def scheduler_event(self, data: st.DataObject, event: str) -> None:
        fake = self.world.fake
        job = data.draw(st.sampled_from([job for job in fake.jobs if unfinished(job)]))
        self.world.clock.advance(seconds=30)
        if event == "START":
            fake.start(job.job)
        elif event == "FORGET":
            fake.forget(job.job)
        elif event == "REQUEUE":
            fake.requeue(job.job)
        else:
            fake.finish(job.job, event, exit_code="0:0" if event == "COMPLETED" else "1:0")

    # --- invariants ---------------------------------------------------------------------------

    @invariant()
    def durable_state_reloads(self) -> None:
        reopened = Campaign.open(self.world.path).status(scheduler=False)
        assert reopened.revision == self.campaign.status(scheduler=False).revision

    @invariant()
    def at_most_one_unfinished_job_per_task(self) -> None:
        owners = {
            f"servatus-{attempt.allocation_id}": attempt.task_keys
            for attempt in self.campaign.status(scheduler=False).attempts
        }
        live: dict[str, int] = {}
        for job in self.world.fake.jobs:
            if unfinished(job):
                for key in owners.get(job.name, ()):
                    live[key] = live.get(key, 0) + 1
        duplicated = {key for key, count in live.items() if count > 1} - self.acknowledged
        assert not duplicated, f"duplicate unfinished work for {sorted(duplicated)}"


def test_random_histories_preserve_the_campaign_invariants() -> None:
    stateful.run_state_machine_as_test(  # pyright: ignore[reportUnknownMemberType]
        CampaignMachine,
        settings=settings(
            max_examples=60,
            stateful_step_count=20,
            deadline=None,
            suppress_health_check=[HealthCheck.too_slow],
        ),
    )
