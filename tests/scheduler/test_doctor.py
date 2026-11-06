"""``diagnose`` reads what a profile needs from a cluster and never changes anything there."""

from __future__ import annotations

from typing import Any

import pytest
from support.builders import profile, resources, target

from servatus.campaign import Check, Completed, Diagnosis, Verdict, diagnose
from servatus.errors import ConfigurationError, Unavailable
from servatus.testing import FakeScheduler

CHECKS = (
    "slurm version",
    "squeue",
    "MinJobAge",
    "ProctrackType",
    "work_root",
    "log_root",
    "sbatch --test-only",
)
READ_ONLY = {"ping", "squeue", "scontrol", "sh"}


def config(age: str = "300 sec", track: str = "proctrack/cgroup") -> Completed:
    text = f"MinJobAge = {age}\nProctrackType = {track}\n"
    return Completed(0, text.encode(), b"")


def run(fake: FakeScheduler | None = None) -> Diagnosis:
    return diagnose(profile(), connect=fake or FakeScheduler())


def answer(diagnosis: Diagnosis, name: str) -> Check:
    return next(check for check in diagnosis.checks if check.name == name)


def test_a_capable_cluster_passes_every_check_and_nothing_is_submitted() -> None:
    fake = FakeScheduler()
    diagnosis = diagnose(profile(), connect=fake)
    assert diagnosis.verdict is Verdict.PASS
    assert [check.name for check in diagnosis.checks] == list(CHECKS)
    assert {check.verdict for check in diagnosis.checks} == {Verdict.PASS}
    assert (diagnosis.profile, diagnosis.tasks_per_allocation) == ("test", 4)
    assert diagnosis.slurm_version == "slurm 23.11.4"
    assert fake.jobs == () and fake.files == {}
    sbatch = [
        call for call in fake.calls if call[0].endswith("sbatch") and call[1:] != ("--version",)
    ]
    assert len(sbatch) == 1 and sbatch[0][-1] == "--test-only"
    assert all(call[0].rpartition("/")[2] in READ_ONLY | {"sbatch"} for call in fake.calls)
    assert fake.count("scontrol") == 1


@pytest.mark.parametrize(
    ("version", "verdict"),
    [
        ("slurm 22.05.11", Verdict.FAIL),
        ("slurm 23.01.9", Verdict.FAIL),
        ("slurm-wlm 21.08.5", Verdict.FAIL),
        ("slurm 23.02.0", Verdict.PASS),
        ("slurm-wlm 24.11.5", Verdict.PASS),
        ("slurm 100.2.0", Verdict.PASS),
    ],
)
def test_the_slurm_version_must_support_export_nil(version: str, verdict: Verdict) -> None:
    check = answer(run(FakeScheduler(version=version)), "slurm version")
    assert check.verdict is verdict
    assert ("--export=NIL" in check.detail) is (verdict is Verdict.FAIL)


def test_an_unreadable_version_fails_and_the_other_checks_still_run() -> None:
    diagnosis = run(FakeScheduler(version="something else"))
    assert answer(diagnosis, "slurm version").detail == (
        "cannot read a Slurm version from 'something else'"
    )
    assert diagnosis.verdict is Verdict.FAIL
    assert answer(diagnosis, "sbatch --test-only").verdict is Verdict.PASS


def test_squeue_that_does_not_answer_is_unavailable_and_names_the_scrubbed_environment() -> None:
    fake = FakeScheduler()
    fake.fail_next("squeue", Completed(1, b"", b"squeue: error: no controller\n"))
    diagnosis = run(fake)
    check = answer(diagnosis, "squeue")
    assert check.verdict is Verdict.UNAVAILABLE and diagnosis.verdict is Verdict.UNAVAILABLE
    assert "exit status 1: squeue: error: no controller" in check.detail
    assert "SLURM_CONF" in check.detail


def test_squeue_output_servatus_cannot_read_fails() -> None:
    fake = FakeScheduler()
    fake.fail_next("squeue", Completed(0, b"banner line\n", b""))
    check = answer(run(fake), "squeue")
    assert check.verdict is Verdict.FAIL and "cannot read" in check.detail


@pytest.mark.parametrize(
    ("age", "verdict", "text"),
    [
        ("300 sec", Verdict.PASS, "300 sec"),
        ("120 sec", Verdict.PASS, "120 sec"),
        ("119 sec", Verdict.FAIL, "under the 120 s"),
        ("0 sec", Verdict.FAIL, "under the 120 s"),
        ("soon", Verdict.FAIL, "not seconds"),
    ],
)
def test_min_job_age_must_outlast_a_cached_directory_listing(
    age: str, verdict: Verdict, text: str
) -> None:
    fake = FakeScheduler()
    fake.fail_next("scontrol", config(age=age))
    check = answer(run(fake), "MinJobAge")
    assert check.verdict is verdict and text in check.detail


def test_process_tracking_must_be_cgroup() -> None:
    fake = FakeScheduler()
    fake.fail_next("scontrol", config(track="proctrack/pgid"))
    check = answer(run(fake), "ProctrackType")
    assert check.verdict is Verdict.FAIL
    assert "'proctrack/pgid', not proctrack/cgroup" in check.detail


def test_settings_scontrol_omits_fail() -> None:
    fake = FakeScheduler()
    fake.fail_next("scontrol", Completed(0, b"Other = 1\n", b""))
    diagnosis = run(fake)
    assert answer(diagnosis, "MinJobAge").verdict is Verdict.FAIL
    assert answer(diagnosis, "ProctrackType").verdict is Verdict.FAIL


def test_scontrol_that_does_not_answer_leaves_both_settings_unknown() -> None:
    fake = FakeScheduler()
    fake.fail_next("scontrol", Completed(1, b"", b"scontrol: error: no\n"))
    fake.fail_next("scontrol", Completed(1, b"", b"scontrol: error: no\n"))
    diagnosis = run(fake)
    assert answer(diagnosis, "MinJobAge").verdict is Verdict.UNAVAILABLE
    assert answer(diagnosis, "ProctrackType").verdict is Verdict.UNAVAILABLE


@pytest.mark.parametrize(
    ("position", "name", "path"),
    [(0, "work_root", "/cluster/work/project"), (1, "log_root", "/cluster/logs/project")],
)
def test_roots_must_be_writable_directories(position: int, name: str, path: str) -> None:
    fake = FakeScheduler()
    for index in range(position + 1):
        fake.fail_next("sh", Completed(0 if index < position else 1, b"", b""))
    check = answer(run(fake), name)
    assert check.verdict is Verdict.FAIL
    assert (
        check.detail
        == f"{path} is not a writable directory on the login host; create it with mkdir -p"
    )


def test_a_refused_shape_fails_with_slurms_reason_and_a_failure_beats_unavailable() -> None:
    fake = FakeScheduler()
    fake.fail_next("squeue", Completed(1, b"", b"down\n"))
    fake.fail_next("sbatch", Completed(1, b"", b"sbatch: error: invalid partition\nmore\n"))
    diagnosis = run(fake)
    assert answer(diagnosis, "sbatch --test-only").detail == (
        "Slurm refuses one Task of this profile: sbatch: error: invalid partition"
    )
    assert diagnosis.verdict is Verdict.FAIL


def test_a_silent_refusal_still_reads_clearly() -> None:
    fake = FakeScheduler()
    fake.fail_next("sbatch", Completed(1, b"", b""))
    check = answer(run(fake), "sbatch --test-only")
    assert check.detail == "Slurm refuses one Task of this profile: no reason given"


def test_the_shape_runs_under_the_direct_launcher_too() -> None:
    direct: dict[str, Any] = {"container": None}
    fake = FakeScheduler()
    assert diagnose(profile(target(**direct)), connect=fake).verdict is Verdict.PASS


def test_an_impossible_profile_is_refused_before_any_contact() -> None:
    fake = FakeScheduler()
    with pytest.raises(ConfigurationError, match="exceeds the target's max_cpus"):
        diagnose(profile(resource_value=resources(cpus=1024)), connect=fake)
    assert fake.calls == ()


def test_a_cluster_that_runs_no_slurm_client_is_unavailable_as_a_whole() -> None:
    fake = FakeScheduler()
    fake.fail_next("ping", Completed(127, b"", b"sbatch: not found\n"))
    with pytest.raises(Unavailable, match="exit status 127: sbatch: not found"):
        diagnose(profile(), connect=fake)
    assert fake.count("squeue") == 0
