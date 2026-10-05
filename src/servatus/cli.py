"""The ``servatus`` command line: a thin adapter over the public API.

Exit codes: 0 success, 1 error, 2 usage (argparse), 3 submission interrupted, 75 cluster
unavailable or campaign busy, 130 interrupted. Errors are one ``servatus: error: ...`` line on
stderr. Human output escapes non-printable characters; ``--json`` output is exact.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

from . import __version__
from .campaign import (
    AcceptanceState,
    Campaign,
    Connect,
    Plan,
    Profile,
    Receipt,
    Retry,
    Status,
    SubmitResult,
    Task,
    capacity,
    ping,
    to_document,
)
from .errors import Busy, ConfigurationError, ServatusError, SubmissionInterrupted, Unavailable
from .publication import publish_file

EXIT_ERROR = 1
EXIT_INTERRUPTED = 3
EXIT_UNAVAILABLE = 75
EXIT_INTERRUPT = 130
MAX_INPUT_BYTES = 64 * 1024 * 1024
_TASK_FIELDS = frozenset({"key", "args", "stdin_file", "env"})

Handler = Callable[[argparse.Namespace], int]


# --- Input -----------------------------------------------------------------------------------


def _read_bytes(path: Path, what: str) -> bytes:
    try:
        with path.open("rb") as source:
            data = source.read(MAX_INPUT_BYTES + 1)
    except OSError as error:
        raise ConfigurationError(f"cannot read {what} {path}: {error.strerror or error}") from None
    if len(data) > MAX_INPUT_BYTES:
        raise ConfigurationError(f"{what} {path} exceeds {MAX_INPUT_BYTES} bytes")
    return data


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ConfigurationError(f"duplicate field {name!r}")
        result[name] = value
    return result


def _task(raw: object, base: Path) -> Task:
    if not isinstance(raw, dict):
        raise ConfigurationError("expected an object with key, args, and optional stdin_file, env")
    fields = cast(dict[str, object], raw)
    if unknown := sorted(fields.keys() - _TASK_FIELDS):
        raise ConfigurationError(f"unknown fields: {', '.join(unknown)}")
    if missing := sorted({"key", "args"} - fields.keys()):
        raise ConfigurationError(f"missing fields: {', '.join(missing)}")
    args = fields["args"]
    if not isinstance(args, list):
        raise ConfigurationError("args must be an array of strings")
    stdin = b""
    if "stdin_file" in fields:
        name = fields["stdin_file"]
        if not isinstance(name, str) or not name:
            raise ConfigurationError("stdin_file must be a nonempty path")
        stdin = _read_bytes(base / name, "stdin_file")
    env = fields.get("env", {})
    if not isinstance(env, dict):
        raise ConfigurationError("env must be an object mapping names to strings")
    return Task(
        cast(str, fields["key"]), cast(list[str], args), stdin=stdin, env=cast(dict[str, str], env)
    )


def _read_tasks(path: Path) -> list[Task]:
    """One Task per JSONL line: ``key``, ``args``, optional ``stdin_file`` (relative to the
    file's directory) and ``env``. Lines end at ``\\n`` only; blank lines are ignored."""
    try:
        text = _read_bytes(path, "task file").decode("utf-8")
    except UnicodeDecodeError:
        raise ConfigurationError(f"task file {path} is not UTF-8") from None
    tasks: list[Task] = []
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            tasks.append(_task(json.loads(line, object_pairs_hook=_unique), path.parent))
        except (ValueError, RecursionError, ConfigurationError) as error:
            raise ConfigurationError(f"invalid task file {path} line {number}: {error}") from None
    return tasks


def _open(arguments: argparse.Namespace) -> Campaign:
    return Campaign.open(arguments.state, connect=arguments.connect)


# --- Output ----------------------------------------------------------------------------------


def _emit(document: object) -> None:
    print(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False))


def _safe(text: object) -> str:
    """``text`` for a terminal: every non-printable character is shown as an escape."""
    return "".join(
        character if character.isprintable() else repr(character)[1:-1] for character in str(text)
    )


def _note(message: str) -> None:
    print(f"servatus: {_safe(message)}", file=sys.stderr)


def _keys(keys: Sequence[str]) -> str:
    return ", ".join(map(_safe, keys))


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _shape(item: Any) -> str:
    """The request of a planned allocation or a shape check, for one human line."""
    return (
        f"{item.cpus} CPUs, {item.memory_mib} MiB, {item.gpus} GPUs, {to_document(item.time_limit)}"
    )


def _receipt_line(receipt: Receipt) -> str:
    return f"allocation {receipt.allocation_id}: job {receipt.job} ({_keys(receipt.task_keys)})"


def _roster(arguments: argparse.Namespace, campaign: Campaign, verb: str) -> int:
    status = campaign.status(scheduler=False)
    if arguments.json:
        _emit({"campaign_id": campaign.id, "tasks": len(status.tasks), "sealed": status.sealed})
    else:
        kind = "sealed" if status.sealed else "appendable"
        print(f"{verb} campaign {campaign.id}: {_plural(len(status.tasks), 'Task')}, {kind}")
    return 0


# --- Authoring -------------------------------------------------------------------------------


def _create(arguments: argparse.Namespace) -> int:
    tasks = _read_tasks(arguments.tasks)
    campaign = Campaign.create(
        arguments.state, tasks, appendable=arguments.appendable, connect=arguments.connect
    )
    return _roster(arguments, campaign, "created")


def _ensure(arguments: argparse.Namespace) -> int:
    tasks = _read_tasks(arguments.tasks)
    campaign = Campaign.ensure(
        arguments.state, tasks, appendable=not arguments.sealed, connect=arguments.connect
    )
    return _roster(arguments, campaign, "ensured")


def _append(arguments: argparse.Namespace) -> int:
    tasks = _read_tasks(arguments.tasks)
    campaign = _open(arguments)
    campaign.append(tasks)
    return _roster(arguments, campaign, f"appended {_plural(len(tasks), 'Task')} to")


def _seal(arguments: argparse.Namespace) -> int:
    campaign = _open(arguments)
    campaign.seal()
    return _roster(arguments, campaign, "sealed")


# --- Planning --------------------------------------------------------------------------------


def _plan_document(plan: Plan, scripts: bool, saved: Path | None) -> dict[str, object]:
    document = cast(dict[str, object], json.loads(plan.to_json()))
    allocations: list[dict[str, object]] = []
    for item in plan.allocations:
        entry = cast(dict[str, object], to_document(item))
        if scripts:
            entry["script"] = item.script.decode("utf-8", "replace")
        else:
            del entry["script"], entry["argv"]
        allocations.append(entry)
    document.update(
        allocations=allocations,
        warnings=list(plan.warnings),
        saved=None if saved is None else str(saved),
    )
    return document


def _print_plan(plan: Plan, scripts: bool, saved: Path | None) -> None:
    decision = plan.decision
    print(
        f"plan for campaign {decision.campaign_id} (revision {decision.revision}, "
        f"profile {_safe(decision.profile.label)})"
    )
    print(f"digest: {plan.digest}")
    allocations = _plural(len(plan.allocations), "allocation")
    print(f"selected: {_plural(len(decision.selected), 'Task')} in {allocations}")
    for item in plan.allocations:
        print(f"  {item.allocation_id}  {_plural(len(item.task_keys), 'Task')}, {_shape(item)}")
        print(f"    {_keys(item.task_keys)}")
    if decision.retry:
        print(f"retry: {_keys(decision.retry)}")
    if decision.duplicate_risk:
        print(f"duplicate risk acknowledged: {_keys(decision.duplicate_risk)}")
    print(f"held: {_plural(len(decision.held), 'Task')}")
    reasons: dict[str, list[str]] = {}
    for key, hold in decision.held.items():
        reasons.setdefault(hold.value, []).append(key)
    for reason, keys in reasons.items():
        print(f"  {reason}: {_keys(keys)}")
    if decision.deferred:
        print(f"deferred: {_plural(len(decision.deferred), 'Task')}")
        print(f"  {_keys(decision.deferred)}")
    if scripts:
        for item in plan.allocations:
            print(f"--- allocation {item.allocation_id} (sensitive) ---")
            print(shlex.join(item.argv))
            print(item.script.decode("utf-8", "replace"))
    if saved is None:
        print("not saved; pass --output PLAN.json to save it")
    else:
        print(f"saved: {_safe(saved)}")


def _plan(arguments: argparse.Namespace) -> int:
    campaign = _open(arguments)
    profile = Profile.load(arguments.config, name=arguments.profile)
    plan = campaign.plan(
        profile,
        retry=Retry.FAILED if arguments.retry_failed else tuple(arguments.retry),
        allow_duplicate_risk=tuple(arguments.allow_duplicate_risk),
        only=tuple(arguments.only) if arguments.only else None,
        tasks_per_allocation=arguments.tasks_per_allocation,
    )
    saved = cast(Path | None, arguments.output)
    if saved is not None:
        plan.save(saved)
    if arguments.show_scripts:
        _note("warning: scripts contain Task arguments, environment, and stdin")
    for warning in plan.warnings:
        _note(f"warning: {warning}")
    if arguments.json:
        _emit(_plan_document(plan, arguments.show_scripts, saved))
    else:
        _print_plan(plan, arguments.show_scripts, saved)
    return 0


def _load_plan(arguments: argparse.Namespace) -> tuple[Campaign, Plan]:
    campaign = _open(arguments)
    return campaign, campaign.load_plan(_read_bytes(arguments.plan, "plan"))


def _validate(arguments: argparse.Namespace) -> int:
    campaign, plan = _load_plan(arguments)
    checks = campaign.validate(plan)
    if arguments.json:
        _emit({"checks": [to_document(check) for check in checks]})
    else:
        for check in checks:
            verdict = "accepted" if check.accepted else "REJECTED"
            print(f"{_plural(check.task_count, 'Task')} ({_shape(check)}): {verdict}")
            for line in f"{check.scheduler_stdout}\n{check.scheduler_stderr}".split("\n"):
                if line.strip():
                    print(f"  {_safe(line)}")
        if not checks:
            print("nothing to validate: the plan has no allocations")
    return 0 if all(check.accepted for check in checks) else EXIT_ERROR


# --- Submission ------------------------------------------------------------------------------


def _print_result(result: SubmitResult, arguments: argparse.Namespace) -> None:
    if arguments.json:
        document = cast(dict[str, object], to_document(result))
        _emit({**document, "complete": result.complete})
        return
    for receipt in result.receipts:
        print(f"submitted {_receipt_line(receipt)}")
    state = _safe(arguments.state)
    for item in result.unresolved:
        print(f"UNRESOLVED allocation {item.allocation_id} ({_keys(item.task_keys)})")
        if (job := item.observed_job) is None:
            print(f"  next: servatus reconcile {state} {item.allocation_id}")
        else:
            option = "" if job.cluster is None else f" --cluster {job.cluster}"
            print(f"  Slurm accepted job {job}; record it:")
            print(
                f"  next: servatus mark-accepted {state} {item.allocation_id} {job.job_id}{option}"
            )
    for item in result.unattempted:
        print(f"not attempted: allocation {item.allocation_id} ({_keys(item.task_keys)})")
    if not result.receipts and result.complete:
        print("nothing to submit: the plan has no allocations")


def _submit(arguments: argparse.Namespace) -> int:
    campaign, plan = _load_plan(arguments)
    try:
        result = campaign.submit(plan)
    except SubmissionInterrupted as error:
        _print_result(error.result, arguments)
        raise
    _print_result(result, arguments)
    return 0


# --- Status and logs -------------------------------------------------------------------------


def _print_status(status: Status, path: Path) -> None:
    state = _safe(path)
    kind = "sealed" if status.sealed else "appendable"
    observed = "observed" if status.scheduler_observed else "not observed (--offline)"
    print(f"campaign {status.campaign_id}: revision {status.revision}, {kind}")
    print(f"at {status.observed_at.isoformat(timespec='seconds')}; scheduler {observed}")
    rows = [("KEY", "RESULT", "EXECUTION", "EXIT", "ALLOCATION")]
    for task in status.tasks:
        if task.unresolved:
            execution = "UNRESOLVED"
        elif task.current_allocation_id is None:
            execution = "-"
        else:
            execution = "ACCEPTED" if task.execution is None else task.execution.value
        allocation = task.current_allocation_id or "-"
        rows.append(
            (_safe(task.key), task.result.value, execution, task.exit_code or "-", allocation)
        )
    widths = [max(len(row[column]) for row in rows) for column in range(4)]
    for row in rows:
        print(
            "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=False))
            + "  "
            + row[4]
        )
    counts = status.counts()
    shown = ", ".join(
        f"{name} {value}" for name, value in counts.items() if value or name == "tasks"
    )
    print(f"counts: {shown}")
    quiescent = "yes" if status.quiescent else "no"
    print(f"quiescent: {quiescent}; results ready: {'yes' if status.results_ready else 'no'}")
    hints = [
        f"servatus reconcile {state} {attempt.allocation_id}"
        for attempt in status.attempts
        if attempt.acceptance is AcceptanceState.UNRESOLVED
    ]
    if counts["failed"] + counts["cancelled"]:
        hints.append(f"servatus plan {state} --retry-failed --output PLAN.json")
    if counts["unsubmitted"]:
        hints.append(f"servatus plan {state} --output PLAN.json")
    if not status.scheduler_observed and status.attempts:
        hints.append(f"servatus status {state}")
    for hint in hints:
        print(f"next: {hint}")


def _status(arguments: argparse.Namespace) -> int:
    status = _open(arguments).status(scheduler=not arguments.offline)
    if arguments.json:
        print(status.to_json().decode("utf-8"))
    else:
        _print_status(status, arguments.state)
    return 0


def _logs(arguments: argparse.Namespace) -> int:
    output = cast(Path | None, arguments.output)
    if output is None and sys.stdout.isatty():
        raise ConfigurationError(
            "refusing to write raw log bytes to a terminal; pass --output FILE or redirect"
        )
    snapshot = _open(arguments).read_log(
        task=arguments.task, allocation=arguments.allocation, max_bytes=arguments.bytes
    )
    content = snapshot.content
    if output is None:
        sys.stdout.buffer.write(content)
        sys.stdout.buffer.flush()
    else:
        publish_file(output, lambda path: path.write_bytes(content), mode=0o600)
    if snapshot.truncated:
        _note(f"showing the last {len(content)} bytes; earlier output exists")
    return 0


# --- Recovery --------------------------------------------------------------------------------


def _print_receipts(arguments: argparse.Namespace, receipts: Sequence[Receipt], verb: str) -> int:
    if arguments.json:
        _emit({verb: [to_document(receipt) for receipt in receipts]})
        return 0
    for receipt in receipts:
        print(f"{verb} {_receipt_line(receipt)}")
    if not receipts:
        print(f"nothing {verb}")
    return 0


def _reconcile(arguments: argparse.Namespace) -> int:
    receipt = _open(arguments).reconcile(arguments.allocation)
    return _print_receipts(arguments, (receipt,), "accepted")


def _mark_accepted(arguments: argparse.Namespace) -> int:
    receipt = _open(arguments).mark_accepted(
        arguments.allocation, arguments.job_id, cluster=arguments.cluster
    )
    return _print_receipts(arguments, (receipt,), "accepted")


def _mark_not_submitted(arguments: argparse.Namespace) -> int:
    _open(arguments).mark_not_submitted(arguments.allocation)
    if arguments.json:
        _emit({"allocation_id": arguments.allocation, "acceptance": "NOT_SUBMITTED"})
    else:
        print(f"allocation {_safe(arguments.allocation)}: recorded as not submitted")
    return 0


def _cancel(arguments: argparse.Namespace) -> int:
    receipts = _open(arguments).cancel(
        tasks=tuple(arguments.task), allocations=tuple(arguments.allocation)
    )
    return _print_receipts(arguments, receipts, "cancelled")


def _doctor(arguments: argparse.Namespace) -> int:
    profile = Profile.load(arguments.config, name=arguments.profile)
    target = profile.target
    fits = capacity(target, profile.resources)
    version = ping(target, connect=arguments.connect)
    if arguments.json:
        _emit(
            {
                "profile": profile.label,
                "host": target.host,
                "slurm_bin": str(target.slurm_bin),
                "sbatch_version": version,
                "tasks_per_allocation": fits,
            }
        )
    else:
        print(f"profile {_safe(profile.label)}: ok, up to {_plural(fits, 'Task')} per allocation")
        print(f"target: {target.host or 'local'} ({target.slurm_bin})")
        print(f"sbatch: {_safe(version)}")
    return 0


# --- Parser ----------------------------------------------------------------------------------

_POSITIONALS: dict[str, dict[str, Any]] = {
    "state": {"type": Path, "metavar": "STATE", "help": "campaign directory"},
    "tasks": {"type": Path, "metavar": "TASKS.jsonl"},
    "plan": {"type": Path, "metavar": "PLAN.json"},
    "allocation": {"metavar": "ALLOCATION"},
    "job_id": {"type": int, "metavar": "JOB_ID"},
}
_KEYS: dict[str, Any] = {"action": "append", "default": [], "metavar": "KEY"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="servatus", description="Durable Slurm campaigns of opaque Tasks."
    )
    parser.add_argument("--version", action="version", version=f"servatus {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    machine = argparse.ArgumentParser(add_help=False)
    machine.add_argument("--json", action="store_true", help="print machine-readable JSON")

    def command(name: str, run: Handler, text: str, *positionals: str) -> argparse.ArgumentParser:
        parents = [] if name == "logs" else [machine]
        sub = commands.add_parser(name, help=text, description=text, parents=parents)
        sub.set_defaults(run=run)
        for positional in positionals:
            sub.add_argument(positional, **_POSITIONALS[positional])
        return sub

    def flag(sub: argparse.ArgumentParser, name: str, text: str) -> None:
        sub.add_argument(name, action="store_true", help=text)

    def profile_options(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--config", type=Path, default=Path("SERVATUS.toml"), metavar="PATH")
        sub.add_argument("--profile", metavar="NAME")

    text = "create a campaign; sealed unless --appendable"
    flag(command("create", _create, text, "state", "tasks"), "--appendable", "allow appending")
    text = "create a campaign, or append Tasks it has not seen"
    flag(command("ensure", _ensure, text, "state", "tasks"), "--sealed", "create it sealed")
    command("append", _append, "append new Tasks to an appendable campaign", "state", "tasks")
    command("seal", _seal, "end authoring irreversibly", "state")
    plan = command("plan", _plan, "show selected, held, and deferred Tasks; save the plan", "state")
    plan.add_argument("--output", type=Path, metavar="PLAN.json", help="save the plan (0600)")
    profile_options(plan)
    retry = plan.add_mutually_exclusive_group()
    retry.add_argument("--retry", **_KEYS)
    retry.add_argument("--retry-failed", action="store_true", help="retry failed or cancelled")
    plan.add_argument("--allow-duplicate-risk", **_KEYS)
    plan.add_argument("--only", **_KEYS)
    plan.add_argument("--tasks-per-allocation", type=int, metavar="N")
    flag(plan, "--show-scripts", "print batch scripts (sensitive)")
    text = "ask sbatch --test-only once per allocation shape"
    command("validate", _validate, text, "state", "plan")
    command("submit", _submit, "submit a reviewed plan", "state", "plan")
    status = command("status", _status, "show Task and allocation status", "state")
    flag(status, "--offline", "do not contact Slurm")
    logs = command("logs", _logs, "write a bounded raw log tail (sensitive)", "state")
    logs.add_argument("--task", metavar="KEY")
    logs.add_argument("--allocation", metavar="ID")
    logs.add_argument("--bytes", type=int, default=65_536, metavar="N")
    logs.add_argument("--output", type=Path, metavar="FILE", help="write a new 0600 file")
    text = "resolve an allocation from Slurm evidence"
    command("reconcile", _reconcile, text, "state", "allocation")
    text = "record a job you found yourself"
    accepted = command("mark-accepted", _mark_accepted, text, "state", "allocation", "job_id")
    accepted.add_argument("--cluster", metavar="NAME")
    text = "record an allocation Slurm proves it never received"
    command("mark-not-submitted", _mark_not_submitted, text, "state", "allocation")
    cancel = command("cancel", _cancel, "scancel the matching accepted allocations", "state")
    cancel.add_argument("--task", **_KEYS)
    cancel.add_argument("--allocation", action="append", default=[], metavar="ID")
    profile_options(command("doctor", _doctor, "check the profile and the scheduler connection"))
    return parser


def main(argv: Sequence[str] | None = None, *, connect: Connect | None = None) -> int:
    """Run one command and return its exit code. ``connect`` replaces the scheduler transport."""
    try:
        arguments = _parser().parse_args(argv)
    except SystemExit as exit_:
        code = exit_.code
        return code if isinstance(code, int) else 0 if code is None else EXIT_ERROR
    arguments.connect = connect
    try:
        return cast(Handler, arguments.run)(arguments)
    except SubmissionInterrupted as error:
        code, message = EXIT_INTERRUPTED, str(error)
    except (Unavailable, Busy) as error:
        code, message = EXIT_UNAVAILABLE, str(error)
    except ServatusError as error:
        code, message = EXIT_ERROR, str(error)
    except OSError as error:
        code, message = EXIT_ERROR, f"{error.strerror or error}"
    except KeyboardInterrupt:
        code, message = EXIT_INTERRUPT, "interrupted"
    except Exception as error:  # a last resort: one line, never a traceback
        code, message = EXIT_ERROR, f"unexpected {type(error).__name__}: {error}"
    print(f"servatus: error: {_safe(message)}", file=sys.stderr)
    return code
