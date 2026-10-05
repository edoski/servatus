"""Slurm operations over one ``Transport``: submit, test, observe, identify, tail, and cancel.

Every method issues fixed, absolute argument vectors and parses replies with the pure functions in
``_evidence``. Command failures raise ``Unavailable``; untrustworthy replies raise
``EvidenceConflict`` (observation) or ``ReconciliationError`` (identification).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import PurePosixPath

from ..errors import ConfigurationError, EvidenceConflict, ReconciliationError, Unavailable
from ._evidence import (
    Expectation,
    JobRef,
    Observation,
    StepEvidence,
    combine,
    is_missing_reply,
    parse_accounting,
    parse_active,
    parse_identity,
    parse_receipt,
    parse_steps,
    submission_window,
)
from ._remote import MAX_STREAM_BYTES, Completed, Transport
from ._script import job_name

MAX_QUERY_JOBS = 16
TAIL = "/usr/bin/tail"
SQUEUE_FORMAT = "--format=%i|%j|%k|%V|%T|%r|%S|%e"
SACCT_ALLOCATION_FORMAT = (
    "--format=JobIDRaw%64,Cluster%256,JobName%256,Comment%256,Submit%32,State%256,ExitCode%32,"
    "Reason%4096,Start%32,End%32"
)
SACCT_STEP_FORMAT = "--format=JobIDRaw%64,JobName%256,State%256,ExitCode%32"
SACCT_IDENTITY_FORMAT = "--format=JobIDRaw,JobName,Comment,Cluster"
_NOTHING_TO_CANCEL = (b"already completing or completed", b"Invalid job id specified")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True, slots=True)
class AttemptQuery:
    """One accepted allocation to observe: its identity, job, intent time, and Task count."""

    allocation_id: str
    job: JobRef
    intent_at: datetime
    task_count: int

    def __post_init__(self) -> None:
        job_name(self.allocation_id)
        if not isinstance(self.job, JobRef):
            raise ConfigurationError("job must be a JobRef")
        if not isinstance(self.intent_at, datetime) or self.intent_at.utcoffset() != timedelta(0):
            raise ConfigurationError("intent_at must be an aware UTC datetime")
        if type(self.task_count) is not int or self.task_count < 1:
            raise ConfigurationError("task_count must be a positive integer")


def _diagnostic(completed: Completed) -> str:
    """The first stderr line, bounded and printable, for operator-facing messages."""
    first = completed.stderr.decode("utf-8", "replace").strip().partition("\n")[0]
    text = _CONTROL.sub("?", first)[:200]
    return f"exit status {completed.returncode}" + (f": {text}" if text else "")


class Scheduler:
    """Slurm commands under ``slurm_bin``, run through ``transport``."""

    def __init__(self, transport: Transport, slurm_bin: PurePosixPath | str) -> None:
        path = PurePosixPath(slurm_bin)
        if not path.is_absolute():
            raise ConfigurationError("slurm_bin must be an absolute POSIX path")
        self._transport = transport
        self._bin = path

    def _command(self, name: str) -> str:
        return str(self._bin / name)

    def _sbatch_request(self, argv: Sequence[str]) -> tuple[str, ...]:
        fields = tuple(argv)
        if not fields or fields[0] != self._command("sbatch") or "--test-only" in fields:
            raise ConfigurationError("submission argv must start with this target's sbatch")
        return fields

    def ping(self) -> str:
        """Prove the transport and Slurm client work: the ``sbatch --version`` text."""
        completed = self._transport.run((self._command("sbatch"), "--version"))
        if completed.returncode != 0:
            raise Unavailable(f"sbatch --version failed with {_diagnostic(completed)}")
        return completed.stdout.decode("utf-8", "replace").strip()

    def submit(self, argv: Sequence[str], script: bytes) -> JobRef:
        """Submit ``script`` with the rendered ``sbatch`` argv and return the accepted job.

        Any failure raises ``Unavailable``; the caller must treat acceptance as unresolved.
        """
        completed = self._transport.run(self._sbatch_request(argv), stdin=script)
        if completed.returncode != 0:
            raise Unavailable(f"sbatch failed with {_diagnostic(completed)}")
        try:
            return parse_receipt(completed.stdout)
        except EvidenceConflict:
            raise Unavailable("sbatch did not print exactly one job identity") from None

    def test_only(self, argv: Sequence[str], script: bytes) -> tuple[bool, str, str]:
        """Ask ``sbatch --test-only`` whether Slurm would accept this request now."""
        completed = self._transport.run((*self._sbatch_request(argv), "--test-only"), stdin=script)
        return (
            completed.returncode == 0,
            completed.stdout.decode("utf-8", "replace").rstrip("\n"),
            completed.stderr.decode("utf-8", "replace").rstrip("\n"),
        )

    def observe(self, queries: Iterable[AttemptQuery]) -> dict[str, Observation]:
        """Observe accepted allocations, keyed by allocation id in query order.

        Queries are grouped by cluster and batched at most 16 jobs at a time; a reused job number
        always goes to a separate batch. Each batch runs ``squeue``, anchored ``sacct``
        allocation history, and a best-effort ``sacct`` step query.
        """
        frozen = tuple(queries)
        if any(not isinstance(query, AttemptQuery) for query in frozen):
            raise ConfigurationError("queries must be AttemptQuery values")
        if len({query.allocation_id for query in frozen}) != len(frozen):
            raise ConfigurationError("each allocation can be observed once per query")
        groups: dict[str | None, list[AttemptQuery]] = {}
        for query in frozen:
            groups.setdefault(query.job.cluster, []).append(query)
        observed: dict[str, Observation] = {}
        for cluster, group in groups.items():
            batch: list[AttemptQuery] = []
            for query in group:
                if len(batch) == MAX_QUERY_JOBS or any(
                    item.job.job_id == query.job.job_id for item in batch
                ):
                    observed.update(self._observe_batch(cluster, batch))
                    batch = []
                batch.append(query)
            if batch:
                observed.update(self._observe_batch(cluster, batch))
        return {query.allocation_id: observed[query.allocation_id] for query in frozen}

    def _observe_batch(
        self, cluster: str | None, batch: Sequence[AttemptQuery]
    ) -> dict[str, Observation]:
        expected: dict[int, Expectation] = {}
        for query in batch:
            start, end = submission_window(query.intent_at)
            expected[query.job.job_id] = Expectation(
                query.allocation_id, start, end, query.task_count
            )
        jobs = ",".join(str(job) for job in expected)
        route = "--local" if cluster is None else f"--clusters={cluster}"
        earliest = min(item.window_start for item in expected.values())
        squeue = self._transport.run(
            (self._command("squeue"), "--noheader", "--jobs", jobs, route, SQUEUE_FORMAT)
        )
        if is_missing_reply(squeue.returncode, squeue.stdout, squeue.stderr, cluster):
            active = parse_active(b"", expected, cluster)
        elif squeue.returncode != 0 or squeue.stderr:
            raise Unavailable(f"squeue failed with {_diagnostic(squeue)}")
        else:
            active = parse_active(squeue.stdout, expected, cluster)
        sacct = self._transport.run(
            (
                self._command("sacct"),
                "--noheader",
                "--parsable2",
                "--allocations",
                "--duplicates",
                "--jobs",
                jobs,
                "--name",
                ",".join(item.identity for item in expected.values()),
                "--starttime",
                earliest,
                route,
                SACCT_ALLOCATION_FORMAT,
            )
        )
        if sacct.returncode != 0 or sacct.stderr:
            raise Unavailable(f"sacct failed with {_diagnostic(sacct)}")
        history = parse_accounting(sacct.stdout, expected, cluster)
        steps = self._steps(expected, jobs, earliest, route)
        return {
            item.allocation_id: Observation(
                combine(item, active[job], history[job]),
                () if steps is None else steps[job],
            )
            for job, item in expected.items()
        }

    def _steps(
        self, expected: dict[int, Expectation], jobs: str, earliest: str, route: str
    ) -> dict[int, tuple[StepEvidence | None, ...]] | None:
        """Per-step evidence, or ``None`` (unavailable) when the step query fails in any way."""
        try:
            completed = self._transport.run(
                (
                    self._command("sacct"),
                    "--noheader",
                    "--parsable2",
                    "--jobs",
                    jobs,
                    "--starttime",
                    earliest,
                    route,
                    SACCT_STEP_FORMAT,
                )
            )
            if completed.returncode != 0 or completed.stderr:
                return None
            return parse_steps(completed.stdout, expected)
        except (Unavailable, EvidenceConflict):
            return None

    def identify(self, allocation_id: str, intent_at: datetime) -> JobRef:
        """The one job Slurm holds for an allocation, else ``ReconciliationError``."""
        jobs = self.find(allocation_id, intent_at)
        if len(jobs) != 1:
            raise ReconciliationError(
                f"scheduler evidence shows {len(jobs)} jobs named {job_name(allocation_id)}, "
                "not exactly one"
            )
        return jobs[0]

    def find(self, allocation_id: str, intent_at: datetime) -> tuple[JobRef, ...]:
        """Every job carrying an allocation's identity: queued or running, or accounted near
        ``intent_at``. Empty proves absence; command failures raise ``Unavailable`` and
        untrustworthy replies ``ReconciliationError``."""
        identity = job_name(allocation_id)
        start, end = submission_window(intent_at)
        squeue = self._transport.run(
            (self._command("squeue"), "--noheader", "--name", identity, "--format=%i|%j|%k")
        )
        if squeue.returncode != 0 or squeue.stderr:
            raise Unavailable(f"squeue failed with {_diagnostic(squeue)}")
        sacct = self._transport.run(
            (
                self._command("sacct"),
                "--noheader",
                "--parsable2",
                "--allocations",
                "--duplicates",
                "--name",
                identity,
                "--starttime",
                start,
                "--endtime",
                end,
                SACCT_IDENTITY_FORMAT,
            )
        )
        if sacct.returncode != 0 or sacct.stderr:
            raise Unavailable(f"sacct failed with {_diagnostic(sacct)}")
        return parse_identity(squeue.stdout, sacct.stdout, identity)

    def tail(self, path: PurePosixPath, max_bytes: int) -> tuple[bytes, bool]:
        """The last ``max_bytes`` of a remote file and whether earlier bytes exist.

        Failures raise ``Unavailable`` without any partial content or remote diagnostics.
        """
        if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_STREAM_BYTES:
            raise ConfigurationError(f"max_bytes must be an integer from 1 to {MAX_STREAM_BYTES}")
        text = str(path)
        if not PurePosixPath(text).is_absolute() or _CONTROL.search(text):
            raise ConfigurationError("log path must be an absolute POSIX path")
        limit = max_bytes + 1
        try:
            completed = self._transport.run((TAIL, "-c", str(limit), "--", text), max_stdout=limit)
        except Unavailable:
            raise Unavailable("log is unavailable") from None
        if completed.returncode != 0 or completed.stderr or len(completed.stdout) > limit:
            raise Unavailable("log is unavailable")
        if len(completed.stdout) == limit:
            return completed.stdout[1:], True
        return completed.stdout, False

    def cancel(self, allocation_id: str, job: JobRef) -> None:
        """Cancel ``job`` only while it still carries this allocation's name.

        A job that already ended or aged out of Slurm counts as cancelled.
        """
        argv = [self._command("scancel")]
        if job.cluster is not None:
            argv.append(f"--clusters={job.cluster}")
        argv.extend((f"--name={job_name(allocation_id)}", str(job.job_id)))
        completed = self._transport.run(argv)
        if completed.returncode != 0 and not any(
            phrase in completed.stderr for phrase in _NOTHING_TO_CANCEL
        ):
            raise Unavailable(f"scancel failed with {_diagnostic(completed)}")
