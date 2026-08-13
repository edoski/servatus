from __future__ import annotations

import copy
import json
from pathlib import Path, PurePosixPath

import pytest
from test_campaign import profile, resources, target, tasks

from servatus import (
    AllocationState,
    AmbiguousSubmission,
    Campaign,
    ConfigurationError,
    PlanError,
    Profile,
    SlurmTarget,
    Task,
    _slurm,
    plan_document,
    restore_plan,
)


def profile_text(*, default: str | None = "cpu", second: bool = False) -> str:
    prefix = "" if default is None else f'default_profile = "{default}"\n'
    document = (
        prefix + "[profiles.cpu.target]\n"
        'host = "login.example.edu"\n'
        'slurm_bin = "/opt/slurm/bin"\n'
        'apptainer = "/usr/bin/apptainer"\n'
        'image = "/images/cpu.sif"\n'
        'work_root = "/work"\n'
        'log_root = "/logs"\n'
        'partitions = ["cpu"]\n'
        "max_tasks_per_allocation = 4\n"
        "max_cpus_per_allocation = 16\n"
        "max_memory_mib_per_allocation = 8192\n"
        "max_gpus_per_allocation = 0\n"
        'max_time_limit = "1-00:00:00"\n'
        "max_allocations_per_submit = 4\n"
        "max_script_bytes = 1048576\n"
        "[profiles.cpu.resources]\n"
        "cpus_per_task = 2\n"
        "memory_mib_per_task = 1024\n"
        "gpus_per_task = 0\n"
        'time_limit = "00:10:00"\n'
    )
    if not second:
        return document
    return document + document.replace(prefix, "").replace("profiles.cpu", "profiles.alias")


def test_profile_loads_the_explicit_complete_lane(tmp_path: Path) -> None:
    path = tmp_path / "SERVATUS.toml"
    path.write_text(profile_text())

    profile = Profile.load(path)

    assert profile.label == "cpu"
    assert profile.target.image == PurePosixPath("/images/cpu.sif")
    assert profile.resources.cpus_per_task == 2


def test_profile_explicit_selection_overrides_default_and_aliases_keep_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "SERVATUS.toml"
    path.write_text(profile_text(second=True))

    selected = Profile.load(path, name="alias")

    assert selected.label == "alias"
    assert selected.target == Profile.load(path).target
    assert selected.resources == Profile.load(path).resources


@pytest.mark.parametrize(
    "contents",
    [
        "",
        "profiles = {}\n",
        profile_text(default=None),
        profile_text(default="missing"),
        "extra = true\n" + profile_text(),
        profile_text().replace("cpus_per_task = 2\n", ""),
        profile_text().replace("[profiles.cpu.resources]", "[profiles.cpu.extra]"),
        profile_text().replace("profiles.cpu", 'profiles.""'),
    ],
)
def test_profile_rejects_missing_selection_or_malformed_document(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "SERVATUS.toml"
    path.write_text(contents)

    with pytest.raises(ConfigurationError):
        Profile.load(path)


def test_profile_validates_malformed_unselected_profile(tmp_path: Path) -> None:
    path = tmp_path / "SERVATUS.toml"
    path.write_text(profile_text() + "\n[profiles.broken]\nvalue = 1\n")

    with pytest.raises(ConfigurationError, match="broken"):
        Profile.load(path, "cpu")


def test_profile_missing_document_and_explicit_label_fail_directly(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="cannot read TOML"):
        Profile.load(tmp_path / "SERVATUS.toml")
    path = tmp_path / "SERVATUS.toml"
    path.write_text(profile_text())
    with pytest.raises(ConfigurationError, match="not declared"):
        Profile.load(path, "missing")


def observations(state: AllocationState):
    def observe(
        _target: SlurmTarget, queries: tuple[_slurm._AttemptQuery, ...]
    ) -> tuple[_slurm.SchedulerObservation, ...]:
        raw = None if state is AllocationState.UNKNOWN else state.value
        return tuple(
            _slurm.SchedulerObservation(state, raw, raw, None, None, None, None)
            for _query in queries
        )

    return observe


def accept(
    monkeypatch: pytest.MonkeyPatch,
    campaign: Campaign,
    *,
    selected_profile: Profile | None = None,
) -> None:
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, b"42;alpha\n", b""),
    )
    campaign.submit(
        campaign.plan(
            selected_profile or profile(),
            view=campaign.inspect(scheduler=False),
        )
    )


@pytest.mark.parametrize("result", [None, False])
def test_never_accepted_missing_or_unobserved_tasks_are_selected(
    tmp_path: Path, result: bool | None
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    view = (
        campaign.inspect(scheduler=False) if result is None else campaign.inspect(lambda _: result)
    )

    plan = campaign.plan(profile(), view=view)

    assert plan.selected_task_keys == ("task-0",)
    assert plan.excluded_task_keys == ()


def test_valid_results_are_always_excluded_and_retry_is_rejected(tmp_path: Path) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    view = campaign.inspect(lambda _: True, scheduler=False)

    assert campaign.plan(profile(), view=view).selected_task_keys == ()
    with pytest.raises(PlanError, match="valid"):
        campaign.plan(profile(), view=view, retry={"task-0"})


@pytest.mark.parametrize("state", [AllocationState.QUEUED, AllocationState.RUNNING])
def test_active_accepted_work_is_withheld_and_cannot_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: AllocationState,
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign)
    monkeypatch.setattr(_slurm, "query_attempts", observations(state))
    view = campaign.inspect()

    assert campaign.plan(profile(), view=view).selected_task_keys == ()
    with pytest.raises(PlanError, match="active"):
        campaign.plan(profile(), view=view, retry={"task-0"})


@pytest.mark.parametrize(
    "state",
    [AllocationState.SUCCEEDED, AllocationState.FAILED, AllocationState.CANCELLED],
)
def test_terminal_accepted_work_requires_explicit_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: AllocationState,
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign)
    monkeypatch.setattr(_slurm, "query_attempts", observations(state))
    view = campaign.inspect()

    assert campaign.plan(profile(), view=view).selected_task_keys == ()
    assert campaign.plan(profile(), view=view, retry={"task-0"}).selected_task_keys == ("task-0",)


def test_unknown_retry_requires_recorded_duplicate_risk_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "campaign"
    campaign = Campaign.open(path, tasks(1))
    accept(monkeypatch, campaign)
    monkeypatch.setattr(_slurm, "query_attempts", observations(AllocationState.UNKNOWN))
    view = campaign.inspect()

    with pytest.raises(PlanError, match="duplicate-risk"):
        campaign.plan(profile(), view=view, retry={"task-0"})
    plan = campaign.plan(
        profile(),
        view=view,
        retry={"task-0"},
        allow_duplicate_risk={"task-0"},
    )
    assert plan.duplicate_risk_task_keys == ("task-0",)
    assert "duplicate execution risk" in plan.warnings[0]

    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, b"43;alpha\n", b""),
    )
    campaign.submit(plan)
    state = json.loads((path / "campaign.json").read_text())
    assert state["attempts"][-1]["duplicate_risk_task_keys"] == ["task-0"]


def test_any_older_active_or_unknown_attempt_governs_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign)
    monkeypatch.setattr(_slurm, "query_attempts", observations(AllocationState.SUCCEEDED))
    retry = campaign.plan(profile(), view=campaign.inspect(), retry={"task-0"})
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, b"43;alpha\n", b""),
    )
    campaign.submit(retry)

    def mixed(
        _target: SlurmTarget, queries: tuple[_slurm._AttemptQuery, ...]
    ) -> tuple[_slurm.SchedulerObservation, ...]:
        states = (AllocationState.QUEUED, AllocationState.SUCCEEDED)
        return tuple(
            _slurm.SchedulerObservation(state, state.value, state.value, None, None, None, None)
            for state, _query in zip(states, queries, strict=True)
        )

    monkeypatch.setattr(_slurm, "query_attempts", mixed)
    with pytest.raises(PlanError, match="active"):
        campaign.plan(profile(), view=campaign.inspect(), retry={"task-0"})

    def older_unknown(
        _target: SlurmTarget, queries: tuple[_slurm._AttemptQuery, ...]
    ) -> tuple[_slurm.SchedulerObservation, ...]:
        states = (AllocationState.UNKNOWN, AllocationState.SUCCEEDED)
        return tuple(
            _slurm.SchedulerObservation(state, None, None, None, None, None, None)
            for state, _query in zip(states, queries, strict=True)
        )

    monkeypatch.setattr(_slurm, "query_attempts", older_unknown)
    view = campaign.inspect()
    with pytest.raises(PlanError, match="duplicate-risk"):
        campaign.plan(profile(), view=view, retry={"task-0"})
    assert campaign.plan(
        profile(),
        view=view,
        retry={"task-0"},
        allow_duplicate_risk={"task-0"},
    ).selected_task_keys == ("task-0",)


def test_ambiguous_allocation_blocks_only_affected_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(2))
    first = campaign.plan(
        profile(target(max_tasks_per_allocation=1)),
        view=campaign.inspect(scheduler=False),
        tasks_per_allocation=1,
    )
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(1, b"", b"lost"),
    )
    with pytest.raises(AmbiguousSubmission):
        campaign.submit(first)

    view = campaign.inspect(scheduler=False)
    plan = campaign.plan(
        profile(target(max_tasks_per_allocation=1)),
        view=view,
        tasks_per_allocation=1,
    )

    assert plan.selected_task_keys == ("task-1",)
    with pytest.raises(PlanError, match="ambiguous"):
        campaign.plan(
            profile(target(max_tasks_per_allocation=1)),
            view=view,
            retry={"task-0"},
        )


def test_accepted_campaign_requires_scheduler_observed_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign)

    with pytest.raises(PlanError, match="scheduler-observed"):
        campaign.plan(profile(), view=campaign.inspect(scheduler=False))


def test_same_value_alias_is_compatible_and_changed_values_fail_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign, selected_profile=profile(label="first"))
    monkeypatch.setattr(_slurm, "query_attempts", observations(AllocationState.SUCCEEDED))
    view = campaign.inspect()

    alias = campaign.plan(profile(label="alias"), view=view, retry={"task-0"})
    assert alias.profile.label == "alias"
    with pytest.raises(PlanError, match="resource semantics"):
        campaign.plan(
            profile(resource_value=resources(cpus_per_task=16), label="first"),
            view=view,
            retry={"task-0"},
        )


def test_foreign_and_stale_views_fail_before_external_contact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = Campaign.open(tmp_path / "first", tasks(1))
    second = Campaign.open(tmp_path / "second", tasks(1))
    foreign = first.inspect(scheduler=False)
    stale = second.inspect(scheduler=False)
    second.seal()
    contacted = False

    def forbidden(*_args: object, **_kwargs: object) -> object:
        nonlocal contacted
        contacted = True
        raise AssertionError

    monkeypatch.setattr(_slurm, "_run_ssh", forbidden)
    with pytest.raises(PlanError, match="another campaign"):
        second.plan(profile(), view=foreign)
    with pytest.raises(PlanError, match="stale"):
        second.plan(profile(), view=stale)
    assert not contacted


def test_plan_round_trip_restores_exact_evidence_without_probe_or_scheduler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    view = campaign.inspect(lambda _: False, scheduler=False)
    plan = campaign.plan(profile(label="lane"), view=view)
    document = plan_document(plan)

    monkeypatch.setattr(
        Campaign,
        "inspect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("probe reran")),
    )
    restored = restore_plan(campaign, copy.deepcopy(document))

    assert plan_document(restored) == document
    assert restored.profile.label == "lane"
    assert restored.selected_task_keys == ("task-0",)


def test_plan_codec_rejects_changed_frozen_view(tmp_path: Path) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    document = plan_document(campaign.plan(profile(), view=campaign.inspect(scheduler=False)))
    changed = copy.deepcopy(document)
    changed["view"]["tasks"][0]["result"] = "VALID"

    with pytest.raises(PlanError):
        restore_plan(campaign, changed)

    malformed_number = copy.deepcopy(document)
    malformed_number["view"]["revision"] = True
    with pytest.raises(PlanError):
        restore_plan(campaign, malformed_number)


def test_restored_plan_label_is_retained_by_submitted_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "campaign"
    campaign = Campaign.open(path, tasks(1))
    plan = campaign.plan(profile(label="restored-lane"), view=campaign.inspect(scheduler=False))
    restored = restore_plan(campaign, plan_document(plan))
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, b"42\n", b""),
    )

    campaign.submit(restored)

    state = json.loads((path / "campaign.json").read_text())
    assert state["attempts"][0]["profile_label"] == "restored-lane"


def test_result_aware_submit_reprobes_selected_tasks_and_aborts_changed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    plan = campaign.plan(profile(), view=campaign.inspect(lambda _: False, scheduler=False))
    contacted = False

    def forbidden(*_args: object, **_kwargs: object) -> object:
        nonlocal contacted
        contacted = True
        raise AssertionError

    monkeypatch.setattr(_slurm, "_run_ssh", forbidden)
    with pytest.raises(PlanError, match="result probe"):
        campaign.submit(plan)
    with pytest.raises(PlanError, match="ineligible"):
        campaign.submit(plan, probe=lambda _: True)
    assert not contacted


def test_scheduler_only_submit_needs_no_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    plan = campaign.plan(profile(), view=campaign.inspect(scheduler=False))
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, b"42\n", b""),
    )

    assert campaign.submit(plan)[0].job_id == 42


def test_submit_rechecks_each_allocation_before_its_sbatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(2))
    planning_probe_calls: list[str] = []

    def planning_probe(task: Task) -> bool:
        planning_probe_calls.append(task.key)
        return False

    view = campaign.inspect(planning_probe, scheduler=False)
    plan = campaign.plan(
        profile(target(max_tasks_per_allocation=1)),
        view=view,
        tasks_per_allocation=1,
    )
    events: list[str] = []

    def submit_probe(task: Task) -> bool:
        events.append(f"probe:{task.key}")
        return False

    def submit(_target: object, _argv: object, _script: object) -> _slurm.Result:
        events.append("sbatch")
        return _slurm.Result(0, f"{40 + events.count('sbatch')}\n".encode(), b"")

    monkeypatch.setattr(_slurm, "_run_ssh", submit)
    campaign.submit(plan, probe=submit_probe)

    assert planning_probe_calls == ["task-0", "task-1"]
    assert events == ["probe:task-0", "sbatch", "probe:task-1", "sbatch"]


def test_submit_does_not_reprobe_valid_excluded_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(2))
    view = campaign.inspect(lambda task: task.key == "task-0", scheduler=False)
    plan = campaign.plan(profile(), view=view)
    probed: list[str] = []

    def probe(task: Task) -> bool:
        probed.append(task.key)
        return False

    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, b"42\n", b""),
    )
    campaign.submit(plan, probe=probe)

    assert plan.excluded_task_keys == ("task-0",)
    assert probed == ["task-1"]


def test_submit_scheduler_refresh_aborts_changed_retry_eligibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign)
    monkeypatch.setattr(_slurm, "query_attempts", observations(AllocationState.SUCCEEDED))
    plan = campaign.plan(profile(), view=campaign.inspect(), retry={"task-0"})
    monkeypatch.setattr(_slurm, "query_attempts", observations(AllocationState.RUNNING))
    contacted = False

    def forbidden(*_args: object, **_kwargs: object) -> object:
        nonlocal contacted
        contacted = True
        raise AssertionError

    monkeypatch.setattr(_slurm, "_run_ssh", forbidden)

    with pytest.raises(PlanError, match="active"):
        campaign.submit(plan)
    assert not contacted


def test_submit_reprobes_then_refreshes_scheduler_and_rereads_before_sbatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign)
    monkeypatch.setattr(_slurm, "query_attempts", observations(AllocationState.SUCCEEDED))
    plan = campaign.plan(
        profile(),
        view=campaign.inspect(lambda _: False),
        retry={"task-0"},
    )
    events: list[str] = []

    def probe(_task: Task) -> bool:
        events.append("probe")
        return False

    def mutate_during_refresh(
        _target: SlurmTarget, queries: tuple[_slurm._AttemptQuery, ...]
    ) -> tuple[_slurm.SchedulerObservation, ...]:
        events.append("scheduler")
        campaign.seal()
        return observations(AllocationState.SUCCEEDED)(_target, queries)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        events.append("sbatch")
        raise AssertionError

    monkeypatch.setattr(_slurm, "query_attempts", mutate_during_refresh)
    monkeypatch.setattr(_slurm, "_run_ssh", forbidden)

    with pytest.raises(PlanError, match="changed during submission freshness"):
        campaign.submit(plan, probe=probe)
    assert events == ["probe", "scheduler"]
