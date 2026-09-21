from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import pytest
from test_campaign import resources, target

from servatus import (
    ConfigurationError,
    Profile,
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


def labeled_profile_text(label: str) -> str:
    encoded = json.dumps(label, ensure_ascii=False)
    return (
        profile_text().replace('"cpu"', encoded, 1).replace("profiles.cpu", f"profiles.{encoded}")
    )


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


@pytest.mark.parametrize("label", ["cpu lane", "計算 🚀", "cpu\tlane"])
def test_profile_labels_are_opaque_nonempty_strings(tmp_path: Path, label: str) -> None:
    path = tmp_path / "SERVATUS.toml"
    path.write_text(labeled_profile_text(label))
    assert Profile.load(path).label == label
    assert Profile.load(path, name=label).label == label


@pytest.mark.parametrize("label", ["", 7])
def test_profile_rejects_only_empty_or_nonstr_labels(label: object) -> None:
    with pytest.raises(ConfigurationError, match="nonempty string"):
        Profile(label, target(), resources())


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


def test_profile_defers_unselected_semantics(tmp_path: Path) -> None:
    path = tmp_path / "SERVATUS.toml"
    path.write_text(
        profile_text(second=True).replace(
            'profiles.alias.target]\nhost = "login.example.edu"',
            'profiles.alias.target]\nhost = "invalid host"',
        )
    )
    assert Profile.load(path).label == "cpu"
    with pytest.raises(ConfigurationError):
        Profile.load(path, "alias")


def test_profile_missing_document_and_explicit_label_fail_directly(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="cannot read TOML"):
        Profile.load(tmp_path / "SERVATUS.toml")
    path = tmp_path / "SERVATUS.toml"
    path.write_text(profile_text())
    with pytest.raises(ConfigurationError, match="not declared"):
        Profile.load(path, "missing")


@pytest.mark.parametrize("section", ["target", "resources"])
def test_unknown_unselected_keys_stay_errors(tmp_path: Path, section: str) -> None:
    path = tmp_path / "SERVATUS.toml"
    path.write_text(
        profile_text(second=True).replace(
            f"[profiles.alias.{section}]", f"[profiles.alias.{section}]\nmisspelled = 1"
        )
    )
    with pytest.raises(ConfigurationError, match="unknown"):
        Profile.load(path)


def test_saved_plan_roundtrip_requires_no_observation_and_retains_reprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_campaign import profile, tasks

    from servatus import Campaign, PlanError, _slurm, plan_document, restore_plan

    campaign = Campaign.create(tmp_path / "campaign", tasks(3))
    plan = campaign.plan(profile(), lambda task: task.key == "task-0")
    document = json.loads(json.dumps(plan_document(plan)))

    def unexpected(*_args):
        raise AssertionError("restoration must remain local")

    monkeypatch.setattr(_slurm, "query_attempts", unexpected)
    restored = restore_plan(campaign, document)
    assert restored == plan and restored.probe_required
    with pytest.raises(PlanError, match="probe"):
        campaign.submit(restored)
    visited = []

    def probe(task):
        visited.append(task.key)
        return True

    with pytest.raises(PlanError, match="ineligible"):
        campaign.submit(restored, probe=probe)
    assert visited == ["task-1", "task-2"]
    assert campaign.inspect(scheduler=False).attempts == ()


def test_reprobe_and_scheduler_refresh_then_atomic_revision_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_campaign import profile, tasks

    from servatus import Campaign, PlanError, _slurm

    campaign = Campaign.create(tmp_path / "campaign", tasks(1), appendable=True)
    plan = campaign.plan(profile(), lambda _: False)
    contacted = []
    monkeypatch.setattr(_slurm, "_run_ssh", lambda *_: contacted.append(True))

    def probe(_task):
        campaign.append(tasks(2)[1:])
        return False

    with pytest.raises(PlanError, match="changed"):
        campaign.submit(plan, probe=probe)
    assert contacted == []
    assert not campaign.inspect(scheduler=False).attempts


def test_reprobe_change_after_first_acceptance_returns_unattempted_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_campaign import profile, tasks

    from servatus import Campaign, _slurm

    campaign = Campaign.create(tmp_path / "campaign", tasks(2))
    plan = campaign.plan(profile(), lambda _: False, tasks_per_allocation=1)
    monkeypatch.setattr(_slurm, "_run_ssh", lambda *_: _slurm.Result(0, b"42\n", b""))
    result = campaign.submit(plan, probe=lambda task: task.key == "task-1")
    assert len(result.receipts) == 1
    assert result.unresolved == () and result.stop_reason
    assert result.unattempted == plan.allocations[1:]


@pytest.mark.parametrize("retained", [False, True])
def test_all_prior_attempts_govern_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retained: bool
) -> None:
    from test_campaign import profile, tasks

    from servatus import Campaign, PlanError, _slurm

    campaign = Campaign.create(tmp_path / "campaign", tasks(1))
    monkeypatch.setattr(_slurm, "_run_ssh", lambda *_: _slurm.Result(0, b"42\n", b""))
    first = campaign.submit(campaign.plan(profile())).receipts[0]

    def observe(_target, queries):
        return tuple(
            _slurm.SchedulerObservation(
                _slurm.AllocationState.UNKNOWN
                if query.allocation_id == first.allocation_id
                else _slurm.AllocationState.SUCCEEDED,
                None,
                None,
                None,
                None,
                None,
                None,
                retained,
            )
            for query in queries
        )

    monkeypatch.setattr(_slurm, "query_attempts", observe)
    if retained:
        with pytest.raises(PlanError, match="active"):
            campaign.plan(profile(), retry=("task-0",), allow_duplicate_risk=("task-0",))
        assert not campaign.inspect().quiescent
    else:
        with pytest.raises(PlanError, match="acknowledgement"):
            campaign.plan(profile(), retry=("task-0",))
        campaign.submit(
            campaign.plan(profile(), retry=("task-0",), allow_duplicate_risk=("task-0",))
        )
        with pytest.raises(PlanError, match="acknowledgement"):
            campaign.plan(profile(), retry=("task-0",))
        assert not campaign.inspect().quiescent


@pytest.mark.parametrize("changed", ["digest", "revision", "selected_task_keys", "probe_required"])
def test_external_plan_rejects_changed_decision(tmp_path: Path, changed: str) -> None:
    from test_campaign import profile, tasks

    from servatus import Campaign, PlanError, plan_document, restore_plan

    campaign = Campaign.create(tmp_path / "campaign", tasks(1))
    document = plan_document(campaign.plan(profile()))
    document[changed] = {
        "digest": "a" * 64,
        "revision": True,
        "selected_task_keys": ["foreign"],
        "probe_required": "yes",
    }[changed]
    with pytest.raises(PlanError):
        restore_plan(campaign, document)


def test_task_key_contract_roundtrips_and_rejects_nul(tmp_path: Path) -> None:
    from test_campaign import profile

    from servatus import Campaign, Task, plan_document, restore_plan

    campaign = Campaign.create(tmp_path / "campaign", (Task("line\n計算", (), b"\0\xff"),))
    plan = campaign.plan(profile())
    assert restore_plan(campaign, plan_document(plan)) == plan
    with pytest.raises(ConfigurationError, match="NUL"):
        Task("bad\0key", (), b"")


@pytest.mark.parametrize("state", ["SUCCEEDED", "FAILED", "CANCELLED"])
def test_terminal_work_requires_explicit_retry_and_valid_results_are_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    from test_campaign import profile, tasks

    from servatus import Campaign, PlanError, _slurm

    campaign = Campaign.create(tmp_path / "campaign", tasks(1))
    monkeypatch.setattr(_slurm, "_run_ssh", lambda *_: _slurm.Result(0, b"42\n", b""))
    campaign.submit(campaign.plan(profile()))
    monkeypatch.setattr(
        _slurm,
        "query_attempts",
        lambda _route, queries: tuple(
            _slurm.SchedulerObservation(
                _slurm.AllocationState(state), None, None, None, None, None, None
            )
            for _ in queries
        ),
    )
    assert campaign.plan(profile()).excluded_task_keys == ("task-0",)
    assert campaign.plan(profile(), retry=("task-0",)).selected_task_keys == ("task-0",)
    assert campaign.plan(profile(), lambda _: True).excluded_task_keys == ("task-0",)
    with pytest.raises(PlanError, match="valid"):
        campaign.plan(profile(), lambda _: True, retry=("task-0",))
