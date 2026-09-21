from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pytest

from servatus import (
    Campaign,
    ConfigurationError,
    PlanError,
    Profile,
    ResourceRequest,
    SlurmTarget,
    Task,
    _campaign,
    _slurm,
)


def target(**changes: object) -> SlurmTarget:
    values: dict[str, object] = {
        "host": "login.example.edu",
        "slurm_bin": PurePosixPath("/opt/slurm/bin"),
        "apptainer": PurePosixPath("/usr/bin/apptainer"),
        "image": PurePosixPath("/cluster/images/work.sif"),
        "work_root": PurePosixPath("/cluster/work/project"),
        "log_root": PurePosixPath("/cluster/logs/project"),
        "partitions": ("gpu",),
        "account": "research",
        "qos": None,
        "constraint": None,
        "gpu_gres": "gpu:a100",
        "max_tasks_per_allocation": 4,
        "max_cpus_per_allocation": 128,
        "max_memory_mib_per_allocation": 262144,
        "max_gpus_per_allocation": 4,
        "max_time_limit": "7-00:00:00",
        "max_allocations_per_submit": 64,
        "max_script_bytes": 4194304,
    }
    values.update(changes)
    return SlurmTarget(**values)


def resources(**changes: object) -> ResourceRequest:
    values: dict[str, object] = {
        "cpus_per_task": 32,
        "memory_mib_per_task": 65536,
        "gpus_per_task": 1,
        "time_limit": "3-00:00:00",
    }
    values.update(changes)
    return ResourceRequest(**values)


def profile(
    target_value: SlurmTarget | None = None,
    resource_value: ResourceRequest | None = None,
    *,
    label: str = "test",
) -> Profile:
    return Profile(label, target_value or target(), resource_value or resources())


def planning(
    campaign: Campaign,
    target_value: SlurmTarget | None = None,
    resource_value: ResourceRequest | None = None,
    **options: object,
) -> _campaign.SubmissionPlan:
    return campaign.plan(profile(target_value, resource_value), **options)


@pytest.fixture(autouse=True)
def terminal_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:

    def observe(
        _target: SlurmTarget, queries: tuple[_slurm._AttemptQuery, ...]
    ) -> tuple[_slurm.SchedulerObservation, ...]:
        return tuple(
            _slurm.SchedulerObservation(
                _slurm.AllocationState.SUCCEEDED,
                "COMPLETED",
                "COMPLETED",
                "0:0",
                None,
                datetime.now(UTC).isoformat(),
                datetime.now(UTC).isoformat(),
            )
            for _query in queries
        )

    monkeypatch.setattr(_slurm, "query_attempts", observe)


def tasks(count: int) -> tuple[Task, ...]:
    return tuple(
        Task(f"task-{index}", ("run", str(index)), f"input-{index}".encode())
        for index in range(count)
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cpus_per_task", True),
        ("cpus_per_task", 0),
        ("memory_mib_per_task", 0),
        ("gpus_per_task", -1),
        ("time_limit", "00:00:00"),
        ("time_limit", "UNLIMITED"),
    ],
)
def test_resource_request_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ConfigurationError):
        resources(**{field: value})


def test_direct_target_and_resources_reject_invalid_values() -> None:
    with pytest.raises(ConfigurationError):
        target(host="-oProxyCommand=bad")
    with pytest.raises(ConfigurationError):
        target(image=PurePosixPath("relative.sif"))


@pytest.mark.parametrize("gres", ["scratch", "gpu:2", "gpu:a100:2", "gpu:a100/bad", "gpu:a100\n"])
def test_gpu_gres_accepts_only_count_free_gpu_family(gres: str) -> None:
    with pytest.raises(ConfigurationError):
        target(gpu_gres=gres)


def test_direct_target_normalizes_pathlike_values_and_rejects_other_types() -> None:
    direct = target(slurm_bin="/opt/slurm/bin", image=Path("/images/work.sif"))
    assert direct.slurm_bin == PurePosixPath("/opt/slurm/bin")
    assert direct.image == PurePosixPath("/images/work.sif")
    assert isinstance(direct.slurm_bin, PurePosixPath)
    with pytest.raises(ConfigurationError):
        target(slurm_bin=7)


def test_kairos_shape_and_balanced_order(tmp_path: Path) -> None:
    campaign = Campaign.create(tmp_path / "campaign", tasks(9), appendable=True)
    plan = planning(campaign, tasks_per_allocation=3)
    assert [len(allocation.task_keys) for allocation in plan.allocations] == [3, 3, 3]
    assert tuple(key for allocation in plan.allocations for key in allocation.task_keys) == tuple(
        task.key for task in tasks(9)
    )
    assert all(
        (allocation.cpus, allocation.memory_mib, allocation.gpus, allocation.time_limit)
        == (96, 196608, 3, "3-00:00:00")
        for allocation in plan.allocations
    )
    four = planning(Campaign.create(tmp_path / "four", tasks(4), appendable=True))
    assert four.allocations[0].cpus == 128
    assert four.allocations[0].memory_mib == 262144
    assert four.allocations[0].gpus == 4
    argv = four.allocations[0].argv
    assert "--nodes=1" in argv
    assert "--ntasks=4" in argv
    assert "--cpus-per-task=32" in argv
    assert "--mem=262144M" in argv
    assert "--gres=gpu:a100:4" in argv
    assert "--time=3-00:00:00" in argv
    assert "--exclusive" not in argv
    script = four.allocations[0].script.decode()
    assert script.count("--cpus-per-task=32") == 4
    assert script.count("--mem=65536M") == 4
    assert script.count("--gres=gpu:a100:1") == 4


@pytest.mark.parametrize(
    "changes",
    [
        {"max_cpus_per_allocation": 31},
        {"max_memory_mib_per_allocation": 65535},
        {"max_gpus_per_allocation": 0, "gpu_gres": None},
        {"max_time_limit": "2-23:58:59"},
    ],
)
def test_every_target_ceiling_can_reject_one_task(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    campaign = Campaign.create(tmp_path / "campaign", tasks(1), appendable=True)
    with pytest.raises(PlanError):
        planning(campaign, target(**changes))


@pytest.mark.parametrize("count", range(1, 18))
@pytest.mark.parametrize("gpus", [0, 1, 2, 4])
def test_balanced_groups_preserve_exact_resources(tmp_path: Path, count: int, gpus: int) -> None:
    maximum = 4
    gpu_ceiling = 0 if gpus == 0 else maximum * gpus
    campaign = Campaign.create(tmp_path / f"campaign-{count}-{gpus}", tasks(count), appendable=True)
    plan = planning(
        campaign,
        target(
            gpu_gres=None if gpus == 0 else "gpu",
            max_gpus_per_allocation=gpu_ceiling,
            max_cpus_per_allocation=maximum * 2,
            max_memory_mib_per_allocation=maximum * 100,
        ),
        resource_value=resources(
            cpus_per_task=2, memory_mib_per_task=100, gpus_per_task=gpus, time_limit="00:10:00"
        ),
    )
    sizes = [len(allocation.task_keys) for allocation in plan.allocations]
    assert sizes == sorted(sizes, reverse=True)
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == count
    assert all(allocation.cpus == len(allocation.task_keys) * 2 for allocation in plan.allocations)
    assert all(
        allocation.memory_mib == len(allocation.task_keys) * 100 for allocation in plan.allocations
    )
    assert all(
        allocation.gpus == len(allocation.task_keys) * gpus for allocation in plan.allocations
    )
    assert all(allocation.time_limit == "00:10:00" for allocation in plan.allocations)


def test_requested_cap_above_feasible_is_rejected(tmp_path: Path) -> None:
    campaign = Campaign.create(tmp_path / "campaign", tasks(4), appendable=True)
    with pytest.raises(PlanError, match="capacity"):
        planning(campaign, target(max_tasks_per_allocation=2), tasks_per_allocation=3)


def test_script_size_exact_boundary(tmp_path: Path) -> None:
    campaign = Campaign.create(tmp_path / "campaign", tasks(1), appendable=True)
    roomy = planning(campaign)
    size = len(roomy.allocations[0].script)
    exact_target = replace(target(), max_script_bytes=size)
    planning(campaign, exact_target)
    with pytest.raises(PlanError, match="script"):
        planning(campaign, replace(exact_target, max_script_bytes=size - 1))


def test_authoring_is_explicit_and_append_accepts_only_new_suffix(tmp_path: Path) -> None:
    from servatus import TaskConflict

    fixed = Campaign.create(tmp_path / "fixed", tasks(1))
    assert fixed.inspect(scheduler=False).sealed
    with pytest.raises(TaskConflict):
        fixed.append(tasks(2)[1:])
    with pytest.raises(TaskConflict):
        Campaign.create(tmp_path / "fixed", tasks(1))
    growing = Campaign.create(tmp_path / "growing", tasks(1), appendable=True)
    plan = planning(growing)
    growing.append(tasks(3)[1:])
    assert Campaign.load(tmp_path / "growing").tasks == tasks(3)
    with pytest.raises(TaskConflict):
        growing.append(tasks(1))
    with pytest.raises(PlanError, match="stale"):
        growing.submit(plan)
    growing.seal()
    before = (tmp_path / "growing" / "campaign.json").read_bytes()
    growing.seal()
    assert (tmp_path / "growing" / "campaign.json").read_bytes() == before


def test_bounded_batches_balance_selected_prefix_and_expose_deferred(tmp_path: Path) -> None:
    campaign = Campaign.create(tmp_path / "campaign", tasks(11))
    plan = planning(campaign, target(max_allocations_per_submit=2), tasks_per_allocation=3)
    assert [len(item.task_keys) for item in plan.allocations] == [3, 3]
    assert plan.selected_task_keys == tuple(task.key for task in tasks(6))
    assert plan.deferred_task_keys == tuple(task.key for task in tasks(11)[6:])
    assert plan.excluded_task_keys == ()


@pytest.mark.parametrize("mutation", ["append", "seal"])
def test_receipt_survives_unrelated_authoring_and_stops_stale_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    from servatus import AcceptanceState

    path = tmp_path / "campaign"
    campaign = Campaign.create(path, tasks(2), appendable=True)
    plan = planning(campaign, tasks_per_allocation=1)
    contacted = []

    def submit(*args):
        durable = Campaign.load(path).inspect(scheduler=False)
        assert durable.attempts[-1].acceptance is AcceptanceState.UNRESOLVED
        contacted.append(args)
        other = Campaign.load(path)
        if mutation == "append":
            other.append(tasks(3)[2:])
        else:
            other.seal()
        return _slurm.Result(0, b"42;alpha\n", b"")

    monkeypatch.setattr(_slurm, "_run_ssh", submit)
    result = campaign.submit(plan)
    assert len(contacted) == len(result.receipts) == 1
    assert result.unresolved == () and result.stop_reason
    assert result.unattempted == plan.allocations[1:]
    recovered = Campaign.load(path).inspect(scheduler=False)
    assert recovered.attempts[0].receipt == result.receipts[0]
    assert recovered.revision == 3


def test_receipt_is_idempotent_but_conflicting_resolution_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from servatus import ReconciliationError

    campaign = Campaign.create(tmp_path / "campaign", tasks(1))
    plan = planning(campaign)
    allocation_id = plan.allocations[0].allocation_id

    def submit(*_args):
        campaign.resolve(allocation_id, job_id=42, cluster="alpha")
        return _slurm.Result(0, b"42;alpha\n", b"")

    monkeypatch.setattr(_slurm, "_run_ssh", submit)
    result = campaign.submit(plan)
    assert len(result.receipts) == 1 and result.stop_reason is None
    before = campaign.inspect(scheduler=False).revision
    campaign.resolve(allocation_id, job_id=42, cluster="alpha")
    assert campaign.inspect(scheduler=False).revision == before
    with pytest.raises(ReconciliationError, match="conflicting"):
        campaign.resolve(allocation_id, job_id=43, cluster="alpha")
    with pytest.raises(ReconciliationError, match="conflicting"):
        campaign.resolve(allocation_id, job_id=None)


@pytest.mark.parametrize("failure", ["transport", "nonzero", "receipt"])
def test_partial_submission_reports_every_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    campaign = Campaign.create(tmp_path / "campaign", tasks(3))
    plan = planning(campaign, tasks_per_allocation=1)
    calls = 0

    def submit(*_args):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _slurm.Result(0, b"42\n", b"")
        if failure == "transport":
            raise OSError("lost transport")
        return _slurm.Result(1 if failure == "nonzero" else 0, b"not a receipt", b"")

    monkeypatch.setattr(_slurm, "_run_ssh", submit)
    result = campaign.submit(plan)
    assert [item.job_id for item in result.receipts] == [42]
    assert result.unresolved[0].allocation_id == plan.allocations[1].allocation_id
    assert result.unresolved[0].observed_receipt is None
    assert result.unattempted == plan.allocations[2:]
    assert result.stop_reason
    assert Campaign.load(tmp_path / "campaign").inspect(scheduler=False).revision == 3
    pending = planning(campaign)
    assert pending.selected_task_keys == ("task-2",)


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_interruption_propagates_with_durable_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: type[BaseException]
) -> None:
    campaign = Campaign.create(tmp_path / "campaign", tasks(1))
    plan = planning(campaign)

    def submit(*_args):
        raise interrupt()

    monkeypatch.setattr(_slurm, "_run_ssh", submit)
    with pytest.raises(interrupt):
        campaign.submit(plan)
    assert campaign.inspect(scheduler=False).attempts[0].acceptance.value == "UNRESOLVED"
    assert not planning(campaign).allocations


def test_observed_receipt_survives_a_failed_durable_receipt_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from servatus import _store

    campaign = Campaign.create(tmp_path / "campaign", tasks(1))
    plan = planning(campaign)
    commit = _store.Transaction.commit

    def failing(tx, state):
        if state.attempts and state.attempts[-1].receipt:
            raise OSError("disk unavailable")
        commit(tx, state)

    monkeypatch.setattr(_store.Transaction, "commit", failing)
    monkeypatch.setattr(_slurm, "_run_ssh", lambda *_: _slurm.Result(0, b"42;alpha\n", b""))
    result = campaign.submit(plan)
    assert result.receipts == ()
    assert result.unresolved[0].observed_receipt.job_id == 42
    assert campaign.inspect(scheduler=False).attempts[0].acceptance.value == "UNRESOLVED"


def test_local_preflight_failure_creates_no_intent(tmp_path: Path) -> None:
    campaign = Campaign.create(tmp_path / "campaign", tasks(1))
    with pytest.raises(PlanError):
        planning(campaign, target(host="x" * 4097))
    assert campaign.inspect(scheduler=False).attempts == ()


def test_retry_can_change_resources_while_history_keeps_original_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    campaign = Campaign.create(tmp_path / "campaign", tasks(1))
    monkeypatch.setattr(_slurm, "_run_ssh", lambda *_: _slurm.Result(0, b"42;alpha\n", b""))
    first = campaign.submit(planning(campaign)).receipts[0]
    calls = []

    def observe(route, queries):
        calls.append((route, queries))
        return tuple(
            _slurm.SchedulerObservation(
                _slurm.AllocationState.FAILED, "FAILED", "FAILED", "1:0", None, None, None
            )
            for _ in queries
        )

    monkeypatch.setattr(_slurm, "query_attempts", observe)
    changed = profile(
        target(host="new.example.edu", log_root=PurePosixPath("/new/logs")),
        replace(resources(), memory_mib_per_task=131072, time_limit="4-00:00:00"),
    )
    second = campaign.submit(campaign.plan(changed, retry=("task-0",))).receipts[0]
    assert all(route.host == "login.example.edu" for route, _ in calls)
    calls.clear()
    assert campaign.inspect().quiescent
    assert {route.host for route, _ in calls} == {"login.example.edu", "new.example.edu"}
    logs = []
    monkeypatch.setattr(_slurm, "read_log_suffix", lambda *args: logs.append(args) or (b"", False))
    campaign.read_log(first.allocation_id)
    campaign.read_log(second.allocation_id)
    assert [args[0].host for args in logs] == ["login.example.edu", "new.example.edu"]
    assert str(logs[1][1]).startswith("/new/logs/")


def test_redacted_record_is_a_simple_current_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from servatus import ObservationError

    campaign = Campaign.create(
        tmp_path / "campaign", (Task("one", ("sensitive",), b"secret"),), appendable=True
    )
    monkeypatch.setattr(_slurm, "_run_ssh", lambda *_: _slurm.Result(0, b"42\n", b""))
    campaign.submit(planning(campaign))
    view = campaign.inspect(lambda _: True)
    record = campaign.record(view)
    assert all(secret not in record for secret in (b"sensitive", b"secret", b"login.example.edu"))
    assert json.loads(record)["attempts"][0]["receipt"]["job_id"] == 42
    campaign.seal()
    with pytest.raises(ObservationError):
        campaign.record(view)


def test_last_receipt_completes_batch_despite_unrelated_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.create(tmp_path / "campaign", tasks(1), appendable=True)

    def submit(*_args):
        campaign.append(tasks(2)[1:])
        return _slurm.Result(0, b"42\n", b"")

    monkeypatch.setattr(_slurm, "_run_ssh", submit)
    result = campaign.submit(planning(campaign))
    assert len(result.receipts) == 1 and result.stop_reason is None
    assert result.unattempted == () and result.unresolved == ()
