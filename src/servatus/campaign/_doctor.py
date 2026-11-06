"""``diagnose``: can this profile run Servatus' protocol on its cluster? Read-only.

Every Slurm fact Servatus relies on is declared in a ``Target`` or assumed by the protocol, never
detected at runtime, so this is where those assumptions meet the real cluster. Nothing here submits
work or writes a file: ``sbatch --test-only`` creates no job.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from pathlib import PurePosixPath

from ..errors import EvidenceConflict, Unavailable
from ._config import Profile, Target, Task
from ._plan import capacity
from ._remote import Transport
from ._remote import connect as connect_target
from ._scheduler import Scheduler
from ._script import render_batch, sbatch_argv

MIN_SLURM = (23, 2)
"""``sbatch --export=NIL`` first exists in Slurm 23.02."""
MIN_JOB_AGE_SECONDS = 120
"""Run markers are read after the purge, which must outlast a cached NFS directory listing."""
_PROBE_ALLOCATION = "0" * 24
_VERSION = re.compile(r"\b([0-9]+)\.([0-9]+)\.[0-9]+")
_SECONDS = re.compile(r"([0-9]+) sec\Z")
_SCRUBBED = "Servatus runs Slurm with only PATH=/usr/bin:/bin, LANG, LC_ALL, and TZ set"


class Verdict(StrEnum):
    """One check's answer. ``UNAVAILABLE``: the cluster did not answer, so nothing is known."""

    PASS = "pass"
    FAIL = "fail"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class Check:
    """One capability of a profile: what was checked, the answer, and what it means."""

    name: str
    verdict: Verdict
    detail: str


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """Every capability check of one profile, whatever the earlier checks answered."""

    profile: str
    host: str | None
    slurm_bin: str
    tasks_per_allocation: int
    slurm_version: str
    checks: tuple[Check, ...]

    @property
    def verdict(self) -> Verdict:
        """``FAIL`` when any check failed, else ``UNAVAILABLE`` when any went unanswered."""
        answers = {check.verdict for check in self.checks}
        for verdict in (Verdict.FAIL, Verdict.UNAVAILABLE):
            if verdict in answers:
                return verdict
        return Verdict.PASS


def _check(name: str, ask: Callable[[], tuple[bool, str]]) -> Check:
    try:
        passed, detail = ask()
    except Unavailable as error:
        return Check(name, Verdict.UNAVAILABLE, str(error))
    return Check(name, Verdict.PASS if passed else Verdict.FAIL, detail)


def _slurm_version(version: str) -> tuple[bool, str]:
    match = _VERSION.search(version)
    if match is None:
        return False, f"cannot read a Slurm version from {version!r}"
    if (int(match.group(1)), int(match.group(2))) < MIN_SLURM:
        return False, (
            f"{version} is older than 23.02: sbatch rejects --export=NIL, which every Servatus "
            "submission uses"
        )
    return True, version


def _squeue(scheduler: Scheduler) -> tuple[bool, str]:
    try:
        scheduler.queue((_PROBE_ALLOCATION,))
    except EvidenceConflict as error:
        return False, f"squeue printed something Servatus cannot read: {error}"
    except Unavailable as error:
        raise Unavailable(f"{error}. {_SCRUBBED}, so SLURM_CONF and modules do not apply") from None
    return True, "answers under the scrubbed environment"


def _min_job_age(config: Mapping[str, str]) -> tuple[bool, str]:
    value = config.get("MinJobAge", "")
    match = _SECONDS.fullmatch(value)
    if match is None:
        return False, f"scontrol show config reports MinJobAge as {value!r}, not seconds"
    if int(match.group(1)) < MIN_JOB_AGE_SECONDS:
        return False, (
            f"MinJobAge is {value}, under the {MIN_JOB_AGE_SECONDS} s run markers rely on: Slurm "
            "must keep a finished job listed longer than a login node caches a directory (ADR 0007)"
        )
    return True, value


def _proctrack(config: Mapping[str, str]) -> tuple[bool, str]:
    value = config.get("ProctrackType", "")
    if value != "proctrack/cgroup":
        return False, (
            f"ProctrackType is {value!r}, not proctrack/cgroup: Servatus treats a job Slurm no "
            "longer lists as gone, and only cgroups guarantee its steps die with it (ADR 0007)"
        )
    return True, value


def _directory(scheduler: Scheduler, path: PurePosixPath) -> tuple[bool, str]:
    if scheduler.writable_directory(path):
        return True, f"{path} is a writable directory"
    return False, f"{path} is not a writable directory on the login host; create it with mkdir -p"


def _one_task_shape(scheduler: Scheduler, profile: Profile) -> tuple[bool, str]:
    target, resources = profile.target, profile.resources
    task = Task("doctor", ("/bin/true",))
    accepted, stdout, stderr = scheduler.test_only(
        sbatch_argv(target, resources, 1, _PROBE_ALLOCATION),
        render_batch(target, resources, (task,), _PROBE_ALLOCATION),
    )
    answer = (stderr or stdout).partition("\n")[0][:200]
    if accepted:
        return True, answer
    return False, f"Slurm refuses one Task of this profile: {answer or 'no reason given'}"


def diagnose(
    profile: Profile, *, connect: Callable[[Target], Transport] | None = None
) -> Diagnosis:
    """Check every capability Servatus needs from ``profile``'s cluster, read-only.

    An impossible profile raises ``ConfigurationError`` before any contact, and a cluster that does
    not run ``sbatch --version`` at all raises ``Unavailable``. Otherwise every check runs and the
    result says what each one found. ``connect`` defaults to ``servatus.campaign.connect``.
    """
    target = profile.target
    fits = capacity(target, profile.resources)
    scheduler = Scheduler((connect or connect_target)(target), target)
    version = scheduler.ping()
    config = cache(scheduler.config)
    checks = (
        _check("slurm version", lambda: _slurm_version(version)),
        _check("squeue", lambda: _squeue(scheduler)),
        _check("MinJobAge", lambda: _min_job_age(config())),
        _check("ProctrackType", lambda: _proctrack(config())),
        _check("work_root", lambda: _directory(scheduler, target.work_root)),
        _check("log_root", lambda: _directory(scheduler, target.log_root)),
        _check("sbatch --test-only", lambda: _one_task_shape(scheduler, profile)),
    )
    return Diagnosis(profile.label, target.host, str(target.slurm_bin), fits, version, checks)
