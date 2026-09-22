from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pytest
from test_campaign import profile, target
from test_planning import profile_text

from servatus import (
    AcceptanceState,
    AllocationEvidence,
    AllocationState,
    AttemptEvidence,
    CampaignView,
    ConfigurationError,
    JobReceipt,
    Profile,
    ResourceRequest,
    ResultState,
    Task,
    TaskEvidence,
)
from servatus._model import profile_document, profile_from_document


def test_task_freezes_argv_and_defaults_to_empty_stdin() -> None:
    args = ["worker", "--flag"]
    task = Task("task", args)
    args.append("changed")
    assert task.args == ("worker", "--flag")
    assert task.stdin == b""
    assert hash(task) == hash(Task("task", ("worker", "--flag")))


@pytest.mark.parametrize("args", ["worker", b"worker", b"", bytearray(), ["worker", 2], {"worker"}])
def test_task_rejects_invalid_argument_sequences(args: object) -> None:
    with pytest.raises(ConfigurationError, match="sequence of strings"):
        Task("task", args)


def test_target_freezes_partition_sequence() -> None:
    partitions = ["cpu"]
    value = target(partitions=partitions)
    partitions.append("changed")
    assert value.partitions == ("cpu",)


def test_minimal_profile_uses_safe_optional_defaults(tmp_path: Path) -> None:
    text = profile_text(default=None)
    for line in (
        "max_allocations_per_submit = 4\n",
        "max_script_bytes = 1048576\n",
        "gpus_per_task = 0\n",
    ):
        text = text.replace(line, "")
    path = tmp_path / "SERVATUS.toml"
    path.write_text(text)
    value = Profile.load(path)
    assert value.label == "cpu"
    assert value.resources == ResourceRequest(
        cpus_per_task=2, memory_mib_per_task=1024, time_limit="00:10:00"
    )
    assert value.target.account is None
    assert value.target.qos is None
    assert value.target.constraint is None
    assert value.target.gpu_gres is None
    assert value.target.max_allocations_per_submit == 1
    assert value.target.max_script_bytes == 1024 * 1024
    assert profile_from_document(profile_document(value)) == value


@pytest.mark.parametrize(
    ("section", "field"),
    [("resources", "gpus_per_task"), ("target", "max_script_bytes")],
)
def test_persisted_profiles_require_fully_resolved_values(section: str, field: str) -> None:
    document = profile_document(profile())
    del document[section][field]
    with pytest.raises(ValueError, match="unexpected object fields"):
        profile_from_document(document)


@pytest.mark.parametrize("path", ["/images//work.sif/", "/images/./work.sif"])
def test_target_normalizes_harmless_path_spelling(path: str) -> None:
    assert target(image=path).image == PurePosixPath("/images/work.sif")


def test_target_preserves_double_slash_root() -> None:
    assert str(target(image="//images//./work.sif").image) == "//images/work.sif"


@pytest.mark.parametrize("path", ["/images/../work.sif", "images/work.sif", "/images/work\n.sif"])
def test_target_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ConfigurationError, match="absolute POSIX path"):
        target(image=path)


def test_snapshot_json_preserves_result_and_scheduler_evidence() -> None:
    now = datetime(2026, 9, 22, tzinfo=UTC)
    view = CampaignView(
        "campaign",
        3,
        True,
        (
            TaskEvidence(
                "task", ResultState.VALID, now, "allocation", AllocationState.FAILED, False
            ),
        ),
        (
            AttemptEvidence(
                "allocation",
                ("task",),
                ("task",),
                ("task",),
                "cpu",
                AcceptanceState.ACCEPTED,
                JobReceipt("allocation", 123, None, ("task",)),
                AllocationEvidence(
                    AllocationState.FAILED,
                    "FAILED",
                    "FAILED",
                    "1:0",
                    "NonZeroExitCode",
                    now.isoformat(),
                    now.isoformat(),
                    now,
                    True,
                ),
            ),
        ),
        True,
        now,
        True,
        False,
    )
    value = json.loads(view.to_json())
    assert value["tasks"][0]["result"] == "VALID"
    assert value["tasks"][0]["result_observed_at"] == now.isoformat()
    assert value["attempts"][0]["receipt"]["job_id"] == 123
    assert value["attempts"][0]["allocation"]["retained"] is True
    assert value["attempts"][0]["allocation"]["observed_at"] == now.isoformat()
    assert value["attempts"][0]["allocation"]["reason"] == "NonZeroExitCode"
    assert value["observed_at"] == now.isoformat()
    assert value["results_ready"] is True
    assert value["quiescent"] is False
