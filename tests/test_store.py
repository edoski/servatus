from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
from test_campaign import planning, tasks

from servatus import Campaign, TaskConflict, _slurm, _store


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "bool_revision",
        "future_revision",
        "duplicate_task",
        "nul_key",
        "payload",
        "unknown_key",
        "reference",
        "intent_revision",
        "outcome_revision",
        "acceptance",
        "capacity",
        "environment",
    ],
)
def test_external_state_is_strict_and_reference_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    path = tmp_path / "campaign"
    campaign = Campaign.create(path, tasks(1))
    monkeypatch.setattr(_slurm, "_run_ssh", lambda *_: _slurm.Result(0, b"42\n", b""))
    campaign.submit(planning(campaign))
    state_path = path / "campaign.json"
    state = json.loads(state_path.read_bytes())
    attempt = state["attempts"][0]
    if mutation == "schema":
        state["schema_version"] = 4
    elif mutation == "bool_revision":
        state["revision"] = True
    elif mutation == "future_revision":
        state["revision"] = 9999
    elif mutation == "duplicate_task":
        state["tasks"].append(state["tasks"][0])
    elif mutation == "nul_key":
        state["tasks"][0]["key"] = "bad\0key"
    elif mutation == "payload":
        state["tasks"][0]["stdin"] = "!"
    elif mutation == "unknown_key":
        attempt["extra"] = 1
    elif mutation == "reference":
        attempt["task_keys"] = ["foreign"]
    elif mutation == "intent_revision":
        attempt["intent_revision"] = 2
    elif mutation == "outcome_revision":
        attempt["outcome_revision"] = 1
    elif mutation == "capacity":
        attempt["profile"]["resources"]["cpus_per_task"] = 1000
    elif mutation == "environment":
        state["tasks"][0]["env"] = {"1A": "x"}
    else:
        attempt["acceptance"] = "NOT_SUBMITTED"
    state_path.write_text(json.dumps(state))
    with pytest.raises(TaskConflict):
        Campaign.load(path)


@pytest.mark.parametrize("content", [b"{", b"[]", b'{"schema_version":5,"schema_version":5}'])
def test_state_rejects_invalid_or_duplicate_json(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "campaign"
    Campaign.create(path, tasks(1))
    (path / "campaign.json").write_bytes(content)
    with pytest.raises(TaskConflict):
        Campaign.load(path)


@pytest.mark.parametrize("entry", ["campaign.json", ".lock", "."])
def test_owner_only_campaign_entries_are_required(tmp_path: Path, entry: str) -> None:
    path = tmp_path / "campaign"
    Campaign.create(path, tasks(1))
    (path / entry).chmod(0o755 if entry == "." else 0o644)
    with pytest.raises(TaskConflict):
        Campaign.load(path)


def test_directory_replacement_and_symlinks_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "campaign"
    campaign = Campaign.create(path, tasks(1))
    path.rename(tmp_path / "original")
    Campaign.create(path, tasks(1))
    with pytest.raises(TaskConflict, match="replaced"):
        campaign.inspect(scheduler=False)
    path.rename(tmp_path / "replacement")
    path.symlink_to(tmp_path / "original", target_is_directory=True)
    with pytest.raises(TaskConflict):
        Campaign.load(path)


@pytest.mark.parametrize("fault", ["write", "file_sync", "replace"])
def test_precommit_failure_preserves_state_and_removes_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    path = tmp_path / "campaign"
    campaign = Campaign.create(path, tasks(1), appendable=True)
    before = (path / "campaign.json").read_bytes()
    sync = os.fsync

    def failed(*_args, **_kwargs):
        raise OSError("injected failure")

    def fsync(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            failed()
        sync(fd)

    monkeypatch.setattr(
        _store.os,
        {"write": "write", "file_sync": "fsync", "replace": "replace"}[fault],
        fsync if fault == "file_sync" else failed,
    )
    with pytest.raises(OSError):
        campaign.append(tasks(2)[1:])
    assert (path / "campaign.json").read_bytes() == before
    assert list(path.glob(".campaign-*.tmp")) == []


def test_postcommit_directory_sync_failure_keeps_recoverable_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "campaign"
    campaign = Campaign.create(path, tasks(1), appendable=True)
    sync = os.fsync

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("injected failure")
        sync(fd)

    monkeypatch.setattr(_store.os, "fsync", fsync)
    with pytest.raises(OSError):
        campaign.append(tasks(2)[1:])
    assert Campaign.load(path).tasks == tasks(2)
    assert list(path.glob(".campaign-*.tmp")) == []


def test_parent_and_intent_are_synced_before_contact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = []
    original = os.fsync

    def fsync(fd):
        events.append(("sync", os.fstat(fd).st_ino))
        original(fd)

    monkeypatch.setattr(_store.os, "fsync", fsync)
    campaign = Campaign.create(tmp_path / "campaign", tasks(1))
    assert events[0] == ("sync", tmp_path.stat().st_ino)
    events.clear()

    def submit(*_args):
        assert len(events) == 2
        assert events[-1] == ("sync", (tmp_path / "campaign").stat().st_ino)
        return _slurm.Result(0, b"42\n", b"")

    monkeypatch.setattr(_slurm, "_run_ssh", submit)
    assert campaign.submit(planning(campaign)).stop_reason is None


def test_state_read_and_write_limits_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "campaign"
    campaign = Campaign.create(path, tasks(1), appendable=True)
    before = (path / "campaign.json").read_bytes()
    monkeypatch.setattr(_store, "_MAX_STATE_BYTES", len(before))
    assert Campaign.load(path).tasks == tasks(1)
    with pytest.raises(TaskConflict):
        campaign.append(tasks(2)[1:])
    assert (path / "campaign.json").read_bytes() == before
    monkeypatch.setattr(_store, "_MAX_STATE_BYTES", len(before) - 1)
    with pytest.raises(TaskConflict, match="large"):
        Campaign.load(path)
