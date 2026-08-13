from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path, PurePosixPath

import pytest
from test_campaign import resources, target, tasks

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
    _slurm,
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
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, f"{next(replies)};alpha\n".encode(), b""),
    )
    while remaining:
        retry_keys = (
            {key for receipt in campaign.status().receipts for key in receipt.task_keys}
            if retry
            else ()
        )
        accepted = campaign.submit(
            campaign.plan(
                target(),
                resources(),
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
        output = b"".join(
            (
                f"{job_id}|alpha|{rows[job_id][0]}|{rows[job_id][1]}|"
                f"{rows[job_id][2]}|{rows[job_id][3]}|{rows[job_id][4]}\n"
            ).encode()
            for job_id in map(int, job_ids.split(","))
            if job_id in rows
        )
        return _slurm.Result(0, output, b"")

    return query


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


def test_exact_receipts_are_queried_in_deterministic_bounded_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(65))
    accept(monkeypatch, campaign, list(range(100, 165)), task_count=1)
    calls: list[tuple[str, ...]] = []

    def query(_target: SlurmTarget, argv: tuple[str, ...]) -> _slurm.Result:
        calls.append(argv)
        return _slurm.Result(0, b"", b"")

    monkeypatch.setattr(_slurm, "_run_bounded_ssh", query)
    view = campaign.inspect()

    assert len(calls) == 4
    assert [argv[0].rsplit("/", 1)[-1] for argv in calls] == [
        "squeue",
        "sacct",
        "squeue",
        "sacct",
    ]
    assert [argv[argv.index("--jobs") + 1] for argv in calls] == [
        ",".join(map(str, range(100, 164))),
        ",".join(map(str, range(100, 164))),
        "164",
        "164",
    ]
    assert all("--clusters=alpha" in argv for argv in calls)
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

    def query(_target: SlurmTarget, argv: tuple[str, ...]) -> _slurm.Result:
        if argv[0].endswith("squeue"):
            return _slurm.Result(
                0,
                b"42|RUNNING|None|2030-01-01T00:00:00|Unknown\n",
                b"",
            )
        return _slurm.Result(
            0,
            b"42|alpha|PENDING|0:0|Priority|Unknown|Unknown\n",
            b"",
        )

    monkeypatch.setattr(_slurm, "_run_bounded_ssh", query)
    view = campaign.inspect(lambda _task: True)
    allocation = view.attempts[0].allocation
    assert allocation is not None
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


def test_conflicting_accounting_rows_or_cluster_identities_abort_complete_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign, [42])

    def conflicting(_target: SlurmTarget, argv: tuple[str, ...]) -> _slurm.Result:
        if argv[0].endswith("squeue"):
            return _slurm.Result(0, b"", b"")
        return _slurm.Result(
            0,
            b"42|alpha|RUNNING|0:0|None|2030-01-01T00:00:00|Unknown\n"
            b"42|alpha|COMPLETED|0:0|None|2030-01-01T00:00:00|2030-01-01T01:00:00\n",
            b"",
        )

    monkeypatch.setattr(_slurm, "_run_bounded_ssh", conflicting)
    with pytest.raises(ObservationError, match="conflicting"):
        campaign.inspect()

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


def test_scheduler_timeout_and_transport_bounds_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign, [42])
    monkeypatch.setattr(
        _slurm,
        "_run_bounded_ssh",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("ssh", 30)),
    )
    with pytest.raises(ObservationError):
        campaign.inspect()

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
        lambda *_args, **_kwargs: _slurm.Result(0, b"x" * (1024 * 1024 + 1), b""),
    )
    with pytest.raises(ObservationError):
        campaign.inspect()

    long_target = target(slurm_bin=PurePosixPath("/" + "x" * 4096))
    oversized = Campaign.open(tmp_path / "oversized", tasks(1))
    monkeypatch.setattr(
        _slurm,
        "_run_ssh",
        lambda *_args, **_kwargs: _slurm.Result(0, b"43;alpha\n", b""),
    )
    oversized.submit(oversized.plan(long_target, resources()))
    monkeypatch.undo()
    with pytest.raises(ObservationError):
        oversized.inspect()


def test_bounded_transport_enforces_deadline_stream_and_command_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign.open(tmp_path / "campaign", tasks(1))
    accept(monkeypatch, campaign, [42])
    real_popen = subprocess.Popen

    def local_process(program: str):
        def start(*_args: object, **kwargs: object) -> subprocess.Popen[bytes]:
            return real_popen([sys.executable, "-c", program], **kwargs)  # type: ignore[arg-type]

        return start

    monkeypatch.setattr(
        _slurm.subprocess,
        "Popen",
        local_process("import time; time.sleep(60)"),
    )
    monkeypatch.setattr(_slurm, "_SSH_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(ObservationError):
        campaign.inspect()

    monkeypatch.setattr(
        _slurm.subprocess,
        "Popen",
        local_process("import os; os.write(1, b'x' * (1024 * 1024 + 1))"),
    )
    monkeypatch.setattr(_slurm, "_SSH_TIMEOUT_SECONDS", 30.0)
    with pytest.raises(ObservationError):
        campaign.inspect()

    for name in ("_MAX_QUERY_ARGC", "_MAX_QUERY_ARG_BYTES"):
        with monkeypatch.context() as bound:
            bound.setattr(_slurm, name, 1)
            with pytest.raises(ObservationError):
                campaign.inspect()

    monkeypatch.setattr(
        _slurm,
        "_run_bounded_ssh",
        lambda *_args, **_kwargs: _slurm.Result(
            0, b"42|alpha|RUNNING||" + b"x" * 4097 + b"|Unknown|Unknown\n", b""
        ),
    )
    with pytest.raises(ObservationError):
        campaign.inspect()


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
        _slurm, "_run_ssh", lambda *_args, **_kwargs: _slurm.Result(1, b"", b"lost")
    )
    with pytest.raises(AmbiguousSubmission):
        campaign.submit(campaign.plan(target(), resources(), retry={"task-0"}))
    monkeypatch.setattr(
        _slurm,
        "_run_bounded_ssh",
        scheduler_rows(
            {41: ("COMPLETED", "0:0", "None", "2030-01-01T00:00:00", "2030-01-01T01:00:00")}
        ),
    )
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
    allocation_id = campaign.status().receipts[0].allocation_id
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
        campaign.submit(campaign.plan(target(), resources()))
    unresolved = campaign.status().ambiguous_allocation_ids[0]
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
    receipt = campaign.submit(campaign.plan(target(), resources()))[0]
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
    first = campaign.submit(campaign.plan(target(), resources()))[0]
    second = campaign.submit(campaign.plan(target(), resources(), retry={"task-0"}))[0]
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
    receipt = campaign.submit(campaign.plan(secret_target, resources()))[0]

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
    first = campaign.submit(campaign.plan(target(), resources()))[0]
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

    extended = Campaign.open(campaign_path, tasks(2))
    extended.plan(target(), resources())
    extended.seal()
    second = extended.submit(extended.plan(target(), resources()))
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
    before_plan = extended.plan(target(), resources())
    extended.read_log(first.allocation_id)
    after_view = extended.inspect(lambda _task: True, scheduler=False)
    after_plan = extended.plan(target(), resources())
    assert state_path.read_bytes() == final_state
    assert (before_view.results_ready, before_view.quiescent) == (
        after_view.results_ready,
        after_view.quiescent,
    )
    assert before_plan == after_plan
