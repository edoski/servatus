from __future__ import annotations

import json
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pytest
from test_campaign import planning, target, tasks

from servatus import (
    AcceptanceState,
    AllocationState,
    AmbiguousSubmission,
    Campaign,
    ConfigurationError,
    LogSnapshot,
    ObservationError,
    ResultState,
    SlurmTarget,
    Task,
    _campaign,
    _slurm,
)


def terminal_observations(
    _target: SlurmTarget, queries: tuple[_slurm._AttemptQuery, ...]
) -> tuple[_slurm.SchedulerObservation, ...]:
    return tuple(
        _slurm.SchedulerObservation(
            _slurm.AllocationState.SUCCEEDED,
            "COMPLETED",
            "COMPLETED",
            "0:0",
            None,
            None,
            None,
        )
        for _query in queries
    )


def accept(
    monkeypatch: pytest.MonkeyPatch,
    campaign: Campaign,
    job_ids: list[int],
    *,
    task_count: int = 1,
    retry: bool = False,
) -> None:
    replies = iter(job_ids)
    remaining = len(job_ids)
    with monkeypatch.context() as context:
        context.setattr(
            _slurm,
            "_run_ssh",
            lambda *_args, **_kwargs: _slurm.Result(0, f"{next(replies)};alpha\n".encode(), b""),
        )
        context.setattr(
            _slurm,
            "query_attempts",
            terminal_observations,
        )
        while remaining:
            evidence = campaign.inspect(scheduler=False)
            retry_keys = (
                {
                    key
                    for attempt in evidence.attempts
                    if attempt.receipt is not None
                    for key in attempt.task_keys
                }
                if retry
                else ()
            )
            accepted = campaign.submit(
                planning(
                    campaign,
                    retry=retry_keys,
                    tasks_per_allocation=task_count,
                )
            )
            assert accepted
            remaining -= len(accepted)


def scheduler_rows(
    rows: dict[int, tuple[str, str, str, str, str]],
):
    def query(_target: SlurmTarget, argv: tuple[str, ...]) -> _slurm.Result:
        job_ids = argv[argv.index("--jobs") + 1]
        if argv[0].endswith("squeue"):
            return _slurm.Result(0, b"", b"")
        identity = argv[argv.index("--name") + 1]
        submitted_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
        output = b"".join(
            accounting_row(identity, submitted_at, *rows[job_id], job_id=job_id)
            for job_id in map(int, job_ids.split(","))
            if job_id in rows
        )
        return _slurm.Result(0, output, b"")

    return query


def accounting_row(
    identity: str,
    submitted_at: str,
    state: str,
    exit_code: str,
    reason: str,
    started_at: str,
    ended_at: str,
    *,
    job_id: int = 42,
    cluster: str = "alpha",
    name: str | None = None,
    comment: str | None = None,
) -> bytes:
    return (
        f"{job_id}|{cluster}|{name or identity}|{comment or identity}|{submitted_at}|{state}|"
        f"{exit_code}|{reason}|{started_at}|{ended_at}\n"
    ).encode()


def active_row(
    identity: str,
    submitted_at: str,
    state: str,
    reason: str = "None",
    started_at: str = "Unknown",
    ended_at: str = "Unknown",
    *,
    name: str | None = None,
    comment: str | None = None,
) -> bytes:
    return (
        f"42|{name or identity}|{comment or identity}|{submitted_at}|{state}|{reason}|"
        f"{started_at}|{ended_at}\n"
    ).encode()


def test_native_missing_active_job_still_uses_exact_terminal_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign, [42])
    receipt = campaign.inspect(scheduler=False).attempts[0].receipt
    identity = f"servatus-{receipt.allocation_id}"
    calls: list[str] = []

    def query(_target: SlurmTarget, argv: tuple[str, ...]) -> _slurm.Result:
        calls.append(argv[0].rsplit("/", 1)[-1])
        if argv[0].endswith("squeue"):
            return _slurm.Result(
                1,
                b"",
                b"slurm_load_jobs error: Invalid job id specified\n",
            )
        submitted_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
        return _slurm.Result(
            0,
            accounting_row(
                identity,
                submitted_at,
                "COMPLETED",
                "0:0",
                "None",
                "2030-01-01T00:00:00",
                "2030-01-01T01:00:00",
            ),
            b"",
        )

    monkeypatch.setattr(_slurm, "_run_bounded_ssh", query)
    evidence = campaign.inspect().attempts[0].allocation

    assert calls == ["squeue", "sacct"]
    assert evidence is not None
    assert evidence.state is AllocationState.SUCCEEDED


@pytest.mark.parametrize(
    "failure",
    [
        _slurm.Result(2, b"", b"slurm_load_jobs error: Invalid job id specified\n"),
        _slurm.Result(1, b"partial", b"slurm_load_jobs error: Invalid job id specified\n"),
        _slurm.Result(1, b"", b"squeue: communication failure\n"),
        _slurm.Result(0, b"", b"slurm_load_jobs error: Invalid job id specified\n"),
    ],
)
def test_only_exact_native_missing_active_shape_is_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: _slurm.Result,
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign, [42])
    monkeypatch.setattr(_slurm, "_run_bounded_ssh", lambda *_args, **_kwargs: failure)

    with pytest.raises(ObservationError):
        campaign.inspect()


def test_reused_job_identity_remains_bound_to_each_immutable_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Clock(datetime):
        current = datetime(2030, 1, 1, 12, 0, 0, tzinfo=UTC)

        @classmethod
        def now(cls, tz: object = None) -> datetime:
            return cls.current

    monkeypatch.setattr(_campaign, "datetime", Clock)
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign, [42])
    Clock.current = datetime(2030, 1, 2, 12, 0, 0, tzinfo=UTC)
    accept(monkeypatch, campaign, [42], retry=True)
    receipts = tuple(
        attempt.receipt
        for attempt in campaign.inspect(scheduler=False).attempts
        if attempt.receipt is not None
    )
    identities = [f"servatus-{receipt.allocation_id}" for receipt in receipts]
    states = {identities[0]: "FAILED", identities[1]: "COMPLETED"}
    submits = {identities[0]: "2030-01-01T12:00:00", identities[1]: "2030-01-02T12:00:00"}
    accounting_calls: list[tuple[str, ...]] = []

    def query(_target: SlurmTarget, argv: tuple[str, ...]) -> _slurm.Result:
        if argv[0].endswith("squeue"):
            return _slurm.Result(1, b"", b"slurm_load_jobs error: Invalid job id specified\n")
        accounting_calls.append(argv)
        identity = argv[argv.index("--name") + 1]
        state = states[identity]
        return _slurm.Result(
            0,
            accounting_row(
                identity,
                submits[identity],
                state,
                "1:0",
                "None",
                "2030-01-01T00:00:00",
                "2030-01-03T00:00:00",
            ),
            b"",
        )

    monkeypatch.setattr(_slurm, "_run_bounded_ssh", query)
    view = campaign.inspect()

    assert [attempt.allocation.state for attempt in view.attempts if attempt.allocation] == [
        AllocationState.FAILED,
        AllocationState.SUCCEEDED,
    ]
    assert [call[call.index("--name") + 1] for call in accounting_calls] == identities
    assert [call[10] for call in accounting_calls] == [
        "2030-01-01T11:00:00",
        "2030-01-02T11:00:00",
    ]


def test_requeue_history_anchors_original_and_uses_unique_latest_incarnation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Clock(datetime):
        @classmethod
        def now(cls, tz: object = None) -> datetime:
            return datetime(2030, 1, 1, 12, 0, 0, tzinfo=UTC)

    monkeypatch.setattr(_campaign, "datetime", Clock)
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign, [42])
    identity = f"servatus-{campaign.inspect(scheduler=False).attempts[0].receipt.allocation_id}"

    def rows(active: bool, terminal: bool = False):
        def query(_target: SlurmTarget, argv: tuple[str, ...]) -> _slurm.Result:
            if argv[0].endswith("squeue"):
                if active:
                    return _slurm.Result(
                        0,
                        active_row(identity, "2030-01-01T15:00:00", "RUNNING"),
                        b"",
                    )
                return _slurm.Result(1, b"", b"slurm_load_jobs error: Invalid job id specified\n")
            latest_state = "COMPLETED" if terminal else "PENDING"
            return _slurm.Result(
                0,
                accounting_row(
                    identity, "2030-01-01T12:00:00", "RUNNING", "0:0", "None", "Unknown", "Unknown"
                )
                + accounting_row(
                    identity,
                    "2030-01-01T14:00:00",
                    latest_state,
                    "0:0",
                    "None",
                    "Unknown",
                    "2030-01-01T16:00:00" if terminal else "Unknown",
                ),
                b"",
            )

        return query

    monkeypatch.setattr(_slurm, "_run_bounded_ssh", rows(active=True))
    active = campaign.inspect().attempts[0].allocation
    assert active is not None
    assert active.state is AllocationState.RUNNING
    assert active.accounting_state == "PENDING"

    monkeypatch.setattr(_slurm, "_run_bounded_ssh", rows(active=False, terminal=True))
    terminal = campaign.inspect().attempts[0].allocation
    assert terminal is not None
    assert terminal.state is AllocationState.SUCCEEDED
    assert terminal.raw_state == "COMPLETED"

    def unanchored(_target: SlurmTarget, argv: tuple[str, ...]) -> _slurm.Result:
        if argv[0].endswith("squeue"):
            return _slurm.Result(
                0,
                active_row(identity, "2030-01-01T15:00:00", "RUNNING"),
                b"",
            )
        return _slurm.Result(0, b"", b"")

    monkeypatch.setattr(_slurm, "_run_bounded_ssh", unanchored)
    unknown = campaign.inspect().attempts[0].allocation
    assert unknown is not None
    assert unknown.state is AllocationState.UNKNOWN


@pytest.mark.parametrize(
    "submissions",
    [
        (("2030-01-01T14:00:00", "RUNNING"),),
        (
            ("2030-01-01T10:00:00", "RUNNING"),
            ("2030-01-01T12:00:00", "RUNNING"),
        ),
        (
            ("2030-01-01T12:00:00", "RUNNING"),
            ("2030-01-01T12:30:00", "PENDING"),
        ),
        (
            ("2030-01-01T12:00:00", "RUNNING"),
            ("2030-01-01T15:00:00", "PENDING"),
            ("2030-01-01T14:00:00", "RUNNING"),
        ),
        (
            ("2030-01-01T12:00:00", "RUNNING"),
            ("2030-01-01T14:00:00", "PENDING"),
            ("2030-01-01T14:00:00", "PENDING"),
        ),
        (
            ("2030-01-01T12:00:00", "RUNNING"),
            ("2030-01-01T14:00:00", "PENDING"),
            ("2030-01-01T14:00:00", "RUNNING"),
        ),
    ],
)
def test_ambiguous_or_noncanonical_requeue_history_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    submissions: tuple[tuple[str, str], ...],
) -> None:
    class Clock(datetime):
        @classmethod
        def now(cls, tz: object = None) -> datetime:
            return datetime(2030, 1, 1, 12, 0, 0, tzinfo=UTC)

    monkeypatch.setattr(_campaign, "datetime", Clock)
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign, [42])
    identity = f"servatus-{campaign.inspect(scheduler=False).attempts[0].receipt.allocation_id}"

    def query(_target: SlurmTarget, argv: tuple[str, ...]) -> _slurm.Result:
        if argv[0].endswith("squeue"):
            return _slurm.Result(0, b"", b"")
        output = b"".join(
            accounting_row(
                identity,
                submitted_at,
                state,
                "0:0",
                "None",
                "Unknown",
                "Unknown",
            )
            for submitted_at, state in submissions
        )
        return _slurm.Result(0, output, b"")

    monkeypatch.setattr(_slurm, "_run_bounded_ssh", query)
    with pytest.raises(ObservationError, match="ambiguous"):
        campaign.inspect()


def test_reused_or_multiply_plausible_accounting_identity_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign, [42])
    receipt = campaign.inspect(scheduler=False).attempts[0].receipt
    identity = f"servatus-{receipt.allocation_id}"
    submitted_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    wrong_identity = "servatus-000000000000000000000000"

    def wrong_active(_target: SlurmTarget, argv: tuple[str, ...]) -> _slurm.Result:
        if argv[0].endswith("squeue"):
            return _slurm.Result(
                0,
                active_row(identity, submitted_at, "RUNNING", name=wrong_identity),
                b"",
            )
        raise AssertionError("unrelated active identity reached accounting")

    monkeypatch.setattr(_slurm, "_run_bounded_ssh", wrong_active)
    with pytest.raises(ObservationError, match="unrelated"):
        campaign.inspect()

    def unrelated(_target: SlurmTarget, argv: tuple[str, ...]) -> _slurm.Result:
        if argv[0].endswith("squeue"):
            return _slurm.Result(0, b"", b"")
        return _slurm.Result(
            0,
            accounting_row(
                identity,
                submitted_at,
                "COMPLETED",
                "0:0",
                "None",
                "2030-01-01T00:00:00",
                "2030-01-01T01:00:00",
                name=wrong_identity,
                comment=wrong_identity,
            ),
            b"",
        )

    monkeypatch.setattr(_slurm, "_run_bounded_ssh", unrelated)
    with pytest.raises(ObservationError, match="unrelated"):
        campaign.inspect()


def test_result_only_inspection_is_read_only_and_derives_sealed_readiness(
    tmp_path: Path,
) -> None:
    plain = tmp_path / "plain.done"
    structured = tmp_path / "structured.json"
    plain.write_text("done\n")
    structured.write_text(json.dumps({"complete": True}))
    tasks = (
        Task("plain", ("ignored", "plain-secret"), b"plain-payload"),
        Task("structured", ("ignored", "json-secret"), b"json-payload"),
    )
    campaign = Campaign.open(tmp_path / "campaign", tasks)
    campaign.seal()
    state_path = tmp_path / "campaign" / "campaign.json"
    before = state_path.read_bytes()

    calls: list[str] = []

    def probe(task: Task) -> bool:
        calls.append(task.key)
        if task.key == "plain":
            return plain.is_file()
        return json.loads(structured.read_text()) == {"complete": True}

    view = campaign.inspect(probe, scheduler=False)

    assert calls == ["plain", "structured"]
    assert view.sealed
    assert view.results_ready
    assert not view.scheduler_observed
    assert not view.quiescent
    assert [task.result for task in view.tasks] == [ResultState.VALID, ResultState.VALID]
    assert state_path.read_bytes() == before
    rendered = repr(view)
    for secret in ("plain-secret", "json-secret", "plain-payload", "json-payload"):
        assert secret not in rendered


def test_probe_states_are_distinct_and_invalid_aborts_without_partial_view(
    tmp_path: Path,
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(3))
    campaign.seal()
    assert [item.result for item in campaign.inspect(scheduler=False).tasks] == [
        ResultState.UNOBSERVED,
        ResultState.UNOBSERVED,
        ResultState.UNOBSERVED,
    ]
    assert [
        item.result
        for item in campaign.inspect(lambda task: task.key == "task-0", scheduler=False).tasks
    ] == [
        ResultState.VALID,
        ResultState.MISSING,
        ResultState.MISSING,
    ]

    calls: list[str] = []

    def invalid(task: Task) -> bool:
        calls.append(task.key)
        if task.key == "task-1":
            raise ValueError("present result is corrupt")
        return True

    with pytest.raises(ValueError, match="corrupt"):
        campaign.inspect(invalid, scheduler=False)
    assert calls == ["task-0", "task-1"]


def test_each_attempt_uses_exact_single_job_scheduler_queries_in_durable_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(3))
    accept(monkeypatch, campaign, [100, 101, 102], task_count=1)
    receipts = tuple(
        attempt.receipt
        for attempt in campaign.inspect(scheduler=False).attempts
        if attempt.receipt is not None
    )
    calls: list[tuple[str, ...]] = []

    def query(_target: SlurmTarget, argv: tuple[str, ...]) -> _slurm.Result:
        calls.append(argv)
        return _slurm.Result(0, b"", b"")

    monkeypatch.setattr(_slurm, "_run_bounded_ssh", query)
    view = campaign.inspect()

    assert len(calls) == 6
    for index, receipt in enumerate(receipts):
        identity = f"servatus-{receipt.allocation_id}"
        queue, accounting = calls[index * 2 : index * 2 + 2]
        assert queue == (
            "/opt/slurm/bin/squeue",
            "--noheader",
            "--jobs",
            str(receipt.job_id),
            "--clusters=alpha",
            "--format=%i|%j|%k|%V|%T|%r|%S|%e",
        )
        assert accounting[:9] == (
            "/opt/slurm/bin/sacct",
            "--noheader",
            "--parsable2",
            "--allocations",
            "--duplicates",
            "--jobs",
            str(receipt.job_id),
            "--name",
            identity,
        )
        assert accounting[9] == "--starttime"
        assert accounting[11:] == (
            "--clusters=alpha",
            "--format=JobIDRaw%64,Cluster%256,JobName%256,Comment%256,Submit%32,"
            "State%256,ExitCode%32,Reason%4096,Start%32,End%32",
        )
    assert {attempt.allocation.state for attempt in view.attempts if attempt.allocation} == {
        AllocationState.UNKNOWN
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        *[
            (state, AllocationState.QUEUED)
            for state in ("PENDING", "CONFIGURING", "REQUEUED", "RESIZING")
        ],
        *[
            (state, AllocationState.RUNNING)
            for state in ("RUNNING", "COMPLETING", "SIGNALING", "STAGE_OUT", "SUSPENDED", "STOPPED")
        ],
        ("COMPLETED", AllocationState.SUCCEEDED),
        *[(state, AllocationState.CANCELLED) for state in ("CANCELLED", "PREEMPTED", "REVOKED")],
        ("CANCELLED by 1234", AllocationState.CANCELLED),
        ("CANCELLED+", AllocationState.CANCELLED),
        *[
            (state, AllocationState.FAILED)
            for state in (
                "BOOT_FAIL",
                "DEADLINE",
                "FAILED",
                "NODE_FAIL",
                "OUT_OF_MEMORY",
                "SPECIAL_EXIT",
                "TIMEOUT",
            )
        ],
        ("FUTURE_STATE", AllocationState.UNKNOWN),
    ],
)
def test_supported_slurm_states_normalize_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: AllocationState,
) -> None:
    campaign = Campaign.open(tmp_path / raw.lower(), tasks(1))
    accept(monkeypatch, campaign, [42])
    monkeypatch.setattr(
        _slurm,
        "_run_bounded_ssh",
        scheduler_rows({42: (raw, "0:0", "reason", "2030-01-01T00:00:00", "2030-01-01T01:00:00")}),
    )

    evidence = campaign.inspect().attempts[0].allocation
    assert evidence is not None
    assert evidence.state is expected
    assert evidence.raw_state == raw


def test_active_scheduler_row_wins_a_legitimate_accounting_transition_and_packed_tasks_share_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(2))
    accept(monkeypatch, campaign, [42], task_count=2)
    campaign.seal()
    receipt = campaign.inspect(scheduler=False).attempts[0].receipt
    identity = f"servatus-{receipt.allocation_id}"
    submitted_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")

    def query(_target: SlurmTarget, argv: tuple[str, ...]) -> _slurm.Result:
        if argv[0].endswith("squeue"):
            return _slurm.Result(
                0,
                active_row(
                    identity,
                    submitted_at,
                    "RUNNING",
                    started_at="2030-01-01T00:00:00",
                ),
                b"",
            )
        return _slurm.Result(
            0,
            accounting_row(
                identity,
                submitted_at,
                "PENDING",
                "0:0",
                "Priority",
                "Unknown",
                "Unknown",
            ),
            b"",
        )

    monkeypatch.setattr(_slurm, "_run_bounded_ssh", query)
    view = campaign.inspect(lambda _task: True)
    allocation = view.attempts[0].allocation
    assert allocation is not None
    assert view.attempts[0].receipt == receipt
    assert allocation.state is AllocationState.RUNNING
    assert allocation.raw_state == "RUNNING"
    assert allocation.accounting_state == "PENDING"
    assert allocation.exit_code == "0:0"
    assert allocation.reason == "Priority"
    assert allocation.started_at == "2030-01-01T00:00:00"
    assert allocation.ended_at is None
    assert [task.execution for task in view.tasks] == [
        AllocationState.RUNNING,
        AllocationState.RUNNING,
    ]
    assert view.results_ready
    assert not view.quiescent
    for secret in (
        "login.example.edu",
        "/cluster/images/work.sif",
        "/cluster/work/project",
        "research",
    ):
        assert secret not in repr(view)


@pytest.mark.parametrize(
    ("active_state", "accounting_state", "expected", "expected_raw", "quiescent"),
    [
        ("RUNNING", "COMPLETED", AllocationState.SUCCEEDED, "COMPLETED", True),
        ("PENDING", "RUNNING", AllocationState.RUNNING, "RUNNING", False),
        ("COMPLETING", "RUNNING", AllocationState.RUNNING, "COMPLETING", False),
        ("CANCELLED by 1234", "CANCELLED+", AllocationState.CANCELLED, "CANCELLED by 1234", True),
        ("COMPLETED", "FAILED", None, None, None),
        ("COMPLETED", "RUNNING", None, None, None),
        ("FAILED", "TIMEOUT", None, None, None),
        ("CANCELLED", "PREEMPTED", None, None, None),
        ("FUTURE_ACTIVE", "FUTURE_ACCOUNTING", None, None, None),
    ],
)
def test_same_incarnation_uses_ordered_later_sample_and_rejects_conflicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    active_state: str,
    accounting_state: str,
    expected: AllocationState | None,
    expected_raw: str | None,
    quiescent: bool | None,
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign, [42])
    receipt = campaign.inspect(scheduler=False).attempts[0].receipt
    identity = f"servatus-{receipt.allocation_id}"
    submitted_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")

    def query(_target: SlurmTarget, argv: tuple[str, ...]) -> _slurm.Result:
        if argv[0].endswith("squeue"):
            return _slurm.Result(
                0,
                active_row(identity, submitted_at, active_state),
                b"",
            )
        return _slurm.Result(
            0,
            accounting_row(
                identity,
                submitted_at,
                accounting_state,
                "0:0",
                "None",
                "2030-01-01T00:00:00",
                "2030-01-01T01:00:00",
            ),
            b"",
        )

    monkeypatch.setattr(_slurm, "_run_bounded_ssh", query)
    if expected is None:
        with pytest.raises(ObservationError, match="conflicting"):
            campaign.inspect()
        return
    view = campaign.inspect()
    allocation = view.attempts[0].allocation
    assert allocation is not None
    assert allocation.state is expected
    assert allocation.raw_state == expected_raw
    assert view.quiescent is quiescent


def test_result_only_view_skips_scheduler_and_valid_results_survive_unknown_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign, [42])
    campaign.seal()
    state_path = tmp_path / "campaign" / "campaign.json"
    before = state_path.read_bytes()

    monkeypatch.setattr(
        _slurm,
        "_run_bounded_ssh",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("scheduler contacted")),
    )
    result_only = campaign.inspect(lambda _task: True, scheduler=False)
    assert result_only.results_ready
    assert not result_only.scheduler_observed
    assert result_only.attempts[0].allocation is None
    assert not result_only.quiescent

    monkeypatch.setattr(
        _slurm,
        "_run_bounded_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, b"", b""),
    )
    observed = campaign.inspect(lambda _task: True)
    assert observed.results_ready
    assert observed.attempts[0].allocation is not None
    assert observed.attempts[0].allocation.state is AllocationState.UNKNOWN
    assert not observed.quiescent
    assert state_path.read_bytes() == before


@pytest.mark.parametrize(
    "failure",
    [
        _slurm.Result(1, b"", b"unavailable"),
        _slurm.Result(0, b"42|alpha|RUNNING\n", b""),
        _slurm.Result(0, b"99|RUNNING|None|Unknown|Unknown\n", b""),
        _slurm.Result(
            0,
            b"42|RUNNING|None|Unknown|Unknown\n42|PENDING|None|Unknown|Unknown\n",
            b"",
        ),
        _slurm.Result(0, b"42|RUNNING|None|Unknown|Unknown\n", b"warning"),
    ],
)
def test_unavailable_malformed_unrelated_conflicting_or_partial_evidence_aborts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: _slurm.Result,
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign, [42])
    monkeypatch.setattr(_slurm, "_run_bounded_ssh", lambda *_args, **_kwargs: failure)
    with pytest.raises(ObservationError):
        campaign.inspect()


@pytest.mark.parametrize("control", [b"\t", b"\r", b"\x00"])
def test_scheduler_fields_reject_controls_before_literal_space_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, control: bytes
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign, [42])
    receipt = campaign.inspect(scheduler=False).attempts[0].receipt
    identity = f"servatus-{receipt.allocation_id}"
    submitted_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")

    monkeypatch.setattr(
        _slurm,
        "_run_bounded_ssh",
        lambda *_args, **_kwargs: _slurm.Result(
            0,
            (f" 42 | {identity} | {identity} | {submitted_at} |RUNNING").encode()
            + control
            + b"|None|Unknown|Unknown\n",
            b"",
        ),
    )

    with pytest.raises(ObservationError, match="malformed"):
        campaign.inspect()


def test_conflicting_accounting_cluster_identity_aborts_complete_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign, [42])

    monkeypatch.setattr(
        _slurm,
        "_run_bounded_ssh",
        scheduler_rows({42: ("COMPLETED", "0:0", "None", "2030-01-01T00:00:00", "Unknown")}),
    )
    original = _slurm._run_bounded_ssh

    def wrong_cluster(target_value: SlurmTarget, argv: tuple[str, ...]) -> _slurm.Result:
        result = original(target_value, argv)
        return _slurm.Result(result.returncode, result.stdout.replace(b"alpha", b"beta"), b"")

    monkeypatch.setattr(_slurm, "_run_bounded_ssh", wrong_cluster)
    with pytest.raises(ObservationError, match="unrelated"):
        campaign.inspect()


def test_scheduler_structural_and_complete_invocation_bounds_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign, [42])
    bounded_ssh = _slurm._run_bounded_ssh
    monkeypatch.setattr(
        _slurm,
        "_run_bounded_ssh",
        lambda *_args, **_kwargs: _slurm.Result(
            0,
            b"42|RUNNING|None|Unknown|Unknown\n" * 129,
            b"",
        ),
    )
    with pytest.raises(ObservationError):
        campaign.inspect()

    monkeypatch.setattr(
        _slurm,
        "_run_bounded_ssh",
        lambda *_args, **_kwargs: _slurm.Result(
            0,
            b"42|alpha|" + b"x" * 4097 + b"|comment|submit|RUNNING|0:0|None|Unknown|Unknown\n",
            b"",
        ),
    )
    with pytest.raises(ObservationError):
        campaign.inspect()

    monkeypatch.setattr(_slurm, "_run_bounded_ssh", bounded_ssh)
    oversized_targets = (
        target(slurm_bin=PurePosixPath("/" + "x" * 4096)),
        target(host="h" * 4097),
    )
    oversized_campaigns: list[Campaign] = []
    for index, oversized_target in enumerate(oversized_targets):
        oversized = Campaign.open(tmp_path / f"oversized-{index}", tasks(1))
        monkeypatch.setattr(
            _slurm,
            "_run_ssh",
            lambda *_args, **_kwargs: _slurm.Result(0, b"43;alpha\n", b""),
        )
        oversized.submit(planning(oversized, oversized_target))
        oversized_campaigns.append(oversized)

    contacted = False

    def forbidden(*_args: object, **_kwargs: object) -> subprocess.Popen[bytes]:
        nonlocal contacted
        contacted = True
        raise AssertionError("oversized SSH invocation reached Popen")

    monkeypatch.setattr(_slurm.subprocess, "Popen", forbidden)
    for oversized in oversized_campaigns:
        with pytest.raises(ObservationError):
            oversized.inspect()
    host_campaign = oversized_campaigns[1]
    with pytest.raises(ObservationError, match="campaign log is unavailable"):
        host_campaign.read_log(
            host_campaign.inspect(scheduler=False).attempts[0].receipt.allocation_id
        )
    assert not contacted


def test_public_inspect_enforces_real_selector_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign, [42])
    real_popen = subprocess.Popen

    def sleeping_process(*_args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        return real_popen(
            [sys.executable, "-c", "import time; time.sleep(0.2)"],
            **kwargs,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(_slurm.subprocess, "Popen", sleeping_process)
    ticks = iter((0.0, 31.0))
    monkeypatch.setattr(_slurm.time, "monotonic", lambda: next(ticks, 31.0))

    with pytest.raises(ObservationError):
        campaign.inspect()


def test_public_inspect_enforces_real_pipe_byte_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign, [42])
    real_popen = subprocess.Popen

    def overflowing_process(*_args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        return real_popen(
            [
                sys.executable,
                "-c",
                "import os; os.write(2, b'x' * 1048577)",
            ],
            **kwargs,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(_slurm.subprocess, "Popen", overflowing_process)

    with pytest.raises(ObservationError, match="byte bound"):
        campaign.inspect()


def test_inspection_and_reconciliation_force_utc_despite_local_timezone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_popen = subprocess.Popen
    real_run_ssh = _slurm._run_ssh
    monkeypatch.setenv("TZ", "Pacific/Honolulu")

    campaign = Campaign.open(tmp_path / "accepted", tasks(1))
    accept(monkeypatch, campaign, [42])
    receipt = campaign.inspect(scheduler=False).attempts[0].receipt
    identity = f"servatus-{receipt.allocation_id}"
    submitted_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    remote_commands: list[str] = []

    def local_scheduler(command: tuple[str, ...], **kwargs: object) -> subprocess.Popen[bytes]:
        remote = command[-1]
        remote_commands.append(remote)
        output = (
            b""
            if "/squeue " in remote
            else accounting_row(
                identity,
                submitted_at,
                "COMPLETED",
                "0:0",
                "None",
                "2030-01-01T00:00:00",
                "2030-01-01T01:00:00",
            )
        )
        return real_popen(
            [sys.executable, "-c", f"import os; os.write(1, {output!r})"],
            **kwargs,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(_slurm.subprocess, "Popen", local_scheduler)
    observed = campaign.inspect().attempts[0].allocation
    assert observed is not None
    assert observed.state is AllocationState.SUCCEEDED

    ambiguous = Campaign.open(tmp_path / "ambiguous", tasks(1))
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(1, b"", b"lost reply"),
    )
    with pytest.raises(AmbiguousSubmission):
        ambiguous.submit(planning(ambiguous))
    allocation_id = ambiguous.inspect(scheduler=False).attempts[-1].allocation_id
    reconcile_identity = f"servatus-{allocation_id}"
    monkeypatch.setattr(_slurm, "_run_ssh", real_run_ssh)

    def local_reconciliation(command: list[str], **kwargs: object) -> _slurm.Result:
        remote = command[-1]
        remote_commands.append(remote)
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert "TZ" not in environment
        if "/squeue " in remote:
            return _slurm.Result(
                0,
                f"77|{reconcile_identity}|{reconcile_identity}\n".encode(),
                b"",
            )
        return _slurm.Result(
            0,
            f"77|{reconcile_identity}|{reconcile_identity}|alpha\n".encode(),
            b"",
        )

    monkeypatch.setattr(_slurm.subprocess, "run", local_reconciliation)
    assert ambiguous.reconcile(allocation_id).job_id == 77
    assert len(remote_commands) == 4
    assert all(
        command.startswith("/usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C TZ=UTC ")
        for command in remote_commands
    )
    assert all("Pacific/Honolulu" not in command for command in remote_commands)


def test_latest_accepted_attempt_projects_current_state_but_quiescence_uses_all_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign, [41])
    accept(monkeypatch, campaign, [42], retry=True)
    monkeypatch.setattr(
        _slurm,
        "_run_bounded_ssh",
        scheduler_rows(
            {
                41: ("RUNNING", "0:0", "None", "2030-01-01T00:00:00", "Unknown"),
                42: ("COMPLETED", "0:0", "None", "2030-01-01T00:00:00", "2030-01-01T01:00:00"),
            }
        ),
    )
    view = campaign.inspect()
    assert len(view.attempts) == 2
    assert view.tasks[0].current_attempt_id == view.attempts[1].allocation_id
    assert view.tasks[0].execution is AllocationState.SUCCEEDED
    assert not view.quiescent

    monkeypatch.setattr(
        _slurm,
        "_run_bounded_ssh",
        scheduler_rows(
            {
                41: (
                    "FAILED",
                    "1:0",
                    "NonZeroExitCode",
                    "2030-01-01T00:00:00",
                    "2030-01-01T01:00:00",
                ),
                42: ("COMPLETED", "0:0", "None", "2030-01-01T00:00:00", "2030-01-01T01:00:00"),
            }
        ),
    )
    assert campaign.inspect().quiescent


def test_unresolved_acceptance_dominates_task_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign, [41])
    monkeypatch.setattr(
        _slurm,
        "_run_bounded_ssh",
        scheduler_rows(
            {41: ("COMPLETED", "0:0", "None", "2030-01-01T00:00:00", "2030-01-01T01:00:00")}
        ),
    )
    monkeypatch.setattr(
        _slurm, "_run_ssh", lambda *_args, **_kwargs: _slurm.Result(1, b"", b"lost")
    )
    with pytest.raises(AmbiguousSubmission):
        campaign.submit(planning(campaign, retry={"task-0"}))
    view = campaign.inspect()
    assert len(view.attempts) == 2
    assert view.tasks[0].acceptance_ambiguous
    assert view.tasks[0].current_attempt_id == view.attempts[1].allocation_id
    assert view.tasks[0].execution is None
    assert not view.quiescent

    campaign.resolve(view.attempts[1].allocation_id, job_id=None)
    resolved = campaign.inspect()
    assert [attempt.acceptance for attempt in resolved.attempts] == [
        AcceptanceState.ACCEPTED,
        AcceptanceState.NOT_SUBMITTED,
    ]
    assert resolved.tasks[0].current_attempt_id == resolved.attempts[0].allocation_id
    assert resolved.tasks[0].execution is AllocationState.SUCCEEDED
    assert resolved.quiescent


def test_campaign_revision_change_after_probe_or_scheduler_observation_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe_campaign = Campaign.open(tmp_path / "probe", tasks(1))

    def mutate_probe(_task: Task) -> bool:
        Campaign.open(tmp_path / "probe", tasks(2))
        return True

    with pytest.raises(ObservationError, match="changed"):
        probe_campaign.inspect(mutate_probe, scheduler=False)

    scheduler_campaign = Campaign.open(tmp_path / "scheduler", tasks(1))
    accept(monkeypatch, scheduler_campaign, [42])
    mutated = False

    def mutate_scheduler(_target: SlurmTarget, _argv: tuple[str, ...]) -> _slurm.Result:
        nonlocal mutated
        if not mutated:
            mutated = True
            scheduler_campaign.seal()
        return _slurm.Result(0, b"", b"")

    monkeypatch.setattr(_slurm, "_run_bounded_ssh", mutate_scheduler)
    with pytest.raises(ObservationError, match="changed"):
        scheduler_campaign.inspect()


def test_accepted_allocation_and_task_logs_are_exact_bounded_binary_suffixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(2))
    accept(monkeypatch, campaign, [42], task_count=2)
    allocation_id = campaign.inspect(scheduler=False).attempts[0].receipt.allocation_id
    replies = iter((b"", b"\x00\xff", b"abcd", b"Xabcd"))
    calls: list[tuple[tuple[str, ...], int]] = []

    def read(
        _target: SlurmTarget,
        argv: tuple[str, ...],
        *,
        max_stdout_bytes: int,
    ) -> _slurm.Result:
        calls.append((argv, max_stdout_bytes))
        return _slurm.Result(0, next(replies), b"")

    monkeypatch.setattr(_slurm, "_run_bounded_ssh", read)
    empty = campaign.read_log(allocation_id, max_bytes=4)
    small = campaign.read_log(allocation_id, task_key="task-1", max_bytes=4)
    exact = campaign.read_log(allocation_id, max_bytes=4)
    over = campaign.read_log(allocation_id, max_bytes=4)

    assert isinstance(empty, LogSnapshot)
    assert (empty.content, empty.truncated) == (b"", False)
    assert (small.content, small.truncated) == (b"\x00\xff", False)
    assert (exact.content, exact.truncated) == (b"abcd", False)
    assert (over.content, over.truncated) == (b"abcd", True)
    assert repr(small) == repr(
        LogSnapshot(b"different-sensitive-content", small.truncated, small.observed_at)
    )
    assert [argv[-1] for argv, _ in calls] == [
        f"/cluster/logs/project/{allocation_id}-42.out",
        f"/cluster/logs/project/{allocation_id}-42-1.out",
        f"/cluster/logs/project/{allocation_id}-42.out",
        f"/cluster/logs/project/{allocation_id}-42.out",
    ]
    assert all(argv[:-1] == ("/usr/bin/tail", "-c", "5", "--") for argv, _ in calls)
    assert [bound for _, bound in calls] == [5, 5, 5, 5]


@pytest.mark.parametrize("maximum", [0, -1, True, 1.5, "1", 1024 * 1024 + 1])
def test_log_byte_limit_is_one_exact_positive_mib_or_less(tmp_path: Path, maximum: object) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    with pytest.raises(ConfigurationError):
        campaign.read_log("foreign", max_bytes=maximum)  # type: ignore[arg-type]


def test_foreign_unaccepted_and_unrelated_log_inputs_fail_before_ssh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(2))
    contacted = False

    def forbidden(*_args: object, **_kwargs: object) -> _slurm.Result:
        nonlocal contacted
        contacted = True
        raise AssertionError("SSH contacted")

    monkeypatch.setattr(_slurm, "_run_bounded_ssh", forbidden)
    with pytest.raises(ObservationError):
        campaign.read_log("foreign")
    assert not contacted

    monkeypatch.setattr(_slurm, "_run_ssh", lambda *_args, **_kwargs: _slurm.Result(1, b"", b""))
    with pytest.raises(AmbiguousSubmission):
        campaign.submit(planning(campaign))
    unresolved = campaign.inspect(scheduler=False).attempts[-1].allocation_id
    with pytest.raises(ObservationError):
        campaign.read_log(unresolved)
    campaign.resolve(unresolved, job_id=None)
    with pytest.raises(ObservationError):
        campaign.read_log(unresolved)
    assert not contacted

    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, b"42;alpha\n", b""),
    )
    receipt = campaign.submit(planning(campaign))[0]
    with pytest.raises(ObservationError):
        campaign.read_log(receipt.allocation_id, task_key="not-in-attempt")
    assert not contacted


def test_retry_logs_with_equal_job_ids_and_distinct_clusters_cannot_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    replies = iter((b"42;alpha\n", b"42;beta\n"))
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, next(replies), b""),
    )
    first = campaign.submit(planning(campaign))[0]
    monkeypatch.setattr(_slurm, "query_attempts", terminal_observations)
    second = campaign.submit(planning(campaign, retry={"task-0"}))[0]
    paths: list[str] = []

    def read(
        _target: SlurmTarget,
        argv: tuple[str, ...],
        *,
        max_stdout_bytes: int,
    ) -> _slurm.Result:
        assert max_stdout_bytes == 65_537
        paths.append(argv[-1])
        return _slurm.Result(0, b"ok", b"")

    monkeypatch.setattr(_slurm, "_run_bounded_ssh", read)
    campaign.read_log(first.allocation_id)
    campaign.read_log(second.allocation_id)

    assert first.job_id == second.job_id == 42
    assert first.cluster == "alpha" and second.cluster == "beta"
    assert first.allocation_id != second.allocation_id
    assert paths == [
        f"/cluster/logs/project/{first.allocation_id}-42.out",
        f"/cluster/logs/project/{second.allocation_id}-42.out",
    ]


@pytest.mark.parametrize(
    "outcome",
    [
        _slurm.Result(1, b"partial-content", b"remote-stderr"),
        _slurm.Result(0, b"partial-content", b"remote-stderr"),
        _slurm.Result(0, b"overflow", b""),
        subprocess.TimeoutExpired("secret-command", 30, output=b"partial-content"),
    ],
)
def test_log_failures_are_redacted_and_never_return_partial_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: _slurm.Result | BaseException,
) -> None:
    secret_target = target(
        host="secret-host",
        log_root=PurePosixPath("/secret/log/path"),
    )
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, b"42;alpha\n", b""),
    )
    receipt = campaign.submit(planning(campaign, secret_target))[0]

    def fail(*_args: object, **_kwargs: object) -> _slurm.Result:
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(_slurm, "_run_bounded_ssh", fail)
    with pytest.raises(ObservationError) as caught:
        campaign.read_log(receipt.allocation_id, max_bytes=4)

    visible: list[str] = []
    error: BaseException | None = caught.value
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        visible.append(str(error))
        visible.extend(getattr(error, "__notes__", ()))
        error = error.__cause__ or error.__context__
    rendered = "\n".join(visible)
    assert rendered == "campaign log is unavailable"
    for secret in (
        "partial-content",
        "remote-stderr",
        "secret-command",
        "secret-host",
        "/secret/log/path",
        receipt.allocation_id,
    ):
        assert secret not in rendered


def test_blocked_log_read_releases_campaign_lock_and_does_not_change_campaign_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_path = tmp_path / "campaign"
    campaign = Campaign.open(campaign_path, tasks(1))
    replies = iter((b"41;alpha\n", b"42;alpha\n"))
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, next(replies), b""),
    )
    first = campaign.submit(planning(campaign))[0]
    state_path = campaign_path / "campaign.json"
    before_log = state_path.read_bytes()
    entered = threading.Event()
    release = threading.Event()

    def blocked(
        _target: SlurmTarget,
        _argv: tuple[str, ...],
        *,
        max_stdout_bytes: int,
    ) -> _slurm.Result:
        assert max_stdout_bytes == 65_537
        entered.set()
        assert release.wait(timeout=5)
        return _slurm.Result(0, b"diagnostic", b"")

    monkeypatch.setattr(_slurm, "_run_bounded_ssh", blocked)
    snapshots: list[LogSnapshot] = []
    failures: list[BaseException] = []

    def read() -> None:
        try:
            snapshots.append(campaign.read_log(first.allocation_id))
        except BaseException as error:
            failures.append(error)

    reader = threading.Thread(target=read)
    reader.start()
    assert entered.wait(timeout=5)

    monkeypatch.setattr(_slurm, "query_attempts", terminal_observations)
    extended = Campaign.open(campaign_path, tasks(2))
    planning(extended)
    extended.seal()
    second = extended.submit(planning(extended))
    assert second[0].task_keys == ("task-1",)

    release.set()
    reader.join(timeout=5)
    assert not reader.is_alive()
    assert failures == []
    assert snapshots[0].content == b"diagnostic"
    assert before_log != state_path.read_bytes()
    final_state = state_path.read_bytes()

    monkeypatch.setattr(
        _slurm,
        "_run_bounded_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, b"read-again", b""),
    )
    before_view = extended.inspect(lambda _task: True, scheduler=False)
    before_plan = planning(extended)
    extended.read_log(first.allocation_id)
    after_view = extended.inspect(lambda _task: True, scheduler=False)
    after_plan = planning(extended)
    assert state_path.read_bytes() == final_state
    assert (before_view.results_ready, before_view.quiescent) == (
        after_view.results_ready,
        after_view.quiescent,
    )
    assert before_plan.selected_task_keys == after_plan.selected_task_keys
    assert before_plan.allocations == after_plan.allocations
