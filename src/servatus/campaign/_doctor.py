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

from ..errors import ConfigurationError, EvidenceConflict, Unavailable
from ._config import Profile, Target, Task
from ._plan import capacity
from ._remote import Transport
from ._remote import connect as connect_target
from ._scheduler import CommandFailed, Scheduler
from ._script import check_command, render_batch, sbatch_argv

MIN_SLURM = (23, 2)
"""Slurm 23.02 first has ``sbatch --export=NIL`` and sets ``SLURM_JOB_END_TIME``, which every
batch script requires."""
MIN_JOB_AGE_SECONDS = 120
"""Run markers are read after the purge, which must outlast a cached NFS directory listing."""
_PROBE_ALLOCATION = "0" * 24
_VERSION = re.compile(r"\b([0-9]+)\.([0-9]+)\.[0-9]+")
_SECONDS = re.compile(r"([0-9]+) sec\Z")
_SCRUBBED = "Servatus runs Slurm with only PATH=/usr/bin:/bin, LANG, LC_ALL, and TZ set"


class Verdict(StrEnum):
    """One check's answer. ``FAIL``: the command ran and said no, or printed something unusable.
    ``UNAVAILABLE``: no answer came back (SSH, a deadline), so nothing is known."""

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
    except CommandFailed as error:
        return Check(name, Verdict.FAIL, str(error))
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
            "submission uses, and jobs lack SLURM_JOB_END_TIME, which every batch script needs"
        )
    return True, version


def _squeue(scheduler: Scheduler) -> tuple[bool, str]:
    try:
        scheduler.queue((_PROBE_ALLOCATION,))
    except EvidenceConflict as error:
        return False, f"squeue printed something Servatus cannot read: {error}"
    except CommandFailed as error:
        return False, f"{error}. {_SCRUBBED}, so SLURM_CONF and modules do not apply"
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
    return False, f"{path} is not a directory this user can write to on the login host"


def _marker_listing(scheduler: Scheduler) -> tuple[bool, str]:
    try:
        scheduler.markers({_PROBE_ALLOCATION: 1})
    except EvidenceConflict as error:
        return False, f"find printed something Servatus cannot read: {error}"
    except CommandFailed as error:
        return False, (
            f"{error}. Servatus reads run markers and their mtimes with find -printf (GNU find)"
        )
    return True, "lists run markers with their mtimes"


def _one_task_shape(scheduler: Scheduler, argv: tuple[str, ...], script: bytes) -> tuple[bool, str]:
    accepted, stdout, stderr = scheduler.test_only(argv, script)
    answer = (stderr or stdout).partition("\n")[0][:200]
    if accepted:
        return True, answer
    return False, f"Slurm refuses one Task of this profile: {answer or 'no reason given'}"


def diagnose(
    profile: Profile, *, connect: Callable[[Target], Transport] | None = None
) -> Diagnosis:
    """Check every capability Servatus needs from ``profile``'s cluster, read-only.

    A profile that cannot render one Task raises ``ConfigurationError`` before any contact.
    Otherwise every check runs and the result says what each one found, except that a cluster
    ``sbatch --version`` gets no answer from stops there: every other command would wait for its
    deadline too. ``connect`` defaults to ``servatus.campaign.connect``.
    """
    if not isinstance(profile, Profile):
        raise ConfigurationError("diagnose needs a Profile")
    target, resources = profile.target, profile.resources
    fits = capacity(target, resources)
    task = Task("doctor", ("/bin/true",))
    argv = check_command(sbatch_argv(target, resources, 1, _PROBE_ALLOCATION))
    script = render_batch(target, resources, (task,), _PROBE_ALLOCATION)
    scheduler = Scheduler((connect or connect_target)(target), target)
    config = cache(scheduler.config)
    later: dict[str, Callable[[], tuple[bool, str]]] = {
        "squeue": lambda: _squeue(scheduler),
        "MinJobAge": lambda: _min_job_age(config()),
        "ProctrackType": lambda: _proctrack(config()),
        "work_root": lambda: _directory(scheduler, target.work_root),
        "log_root": lambda: _directory(scheduler, target.log_root),
        "find -printf": lambda: _marker_listing(scheduler),
        "sbatch --test-only": lambda: _one_task_shape(scheduler, argv, script),
    }
    first = _check("slurm version", lambda: _slurm_version(scheduler.ping()))
    checks = [first]
    if first.verdict is not Verdict.UNAVAILABLE:
        checks.extend(_check(name, ask) for name, ask in later.items())
    return Diagnosis(profile.label, target.host, str(target.slurm_bin), fits, tuple(checks))
