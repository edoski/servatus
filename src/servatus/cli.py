from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from ._campaign import (
    Campaign,
    JobReceipt,
    Profile,
    Task,
    campaign_view_document,
    plan_document,
    restore_plan,
    sensitive_script_document,
    validation_document,
)
from ._errors import ServatusError
from ._workspace import publish_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="servatus", description="Durable Slurm work campaigns")
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="build an evidence-bound immutable plan")
    plan.add_argument("tasks", type=Path)
    plan.add_argument("--campaign", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--profile", metavar="NAME")
    plan.add_argument("--tasks-per-allocation", type=int)
    plan.add_argument(
        "--retry",
        action="append",
        default=[],
        metavar="TASK_KEY",
        help="retry a task with a prior scheduler receipt; repeat for multiple tasks",
    )
    plan.add_argument(
        "--allow-duplicate-risk",
        action="append",
        default=[],
        metavar="TASK_KEY",
        help="acknowledge duplicate-execution risk for an unknown accepted attempt",
    )
    plan.add_argument(
        "--show-scripts",
        action="store_true",
        help="print sensitive complete scripts, including task arguments and payloads",
    )

    validate = commands.add_parser("validate", help="run bounded Slurm test-only validation")
    validate.add_argument("campaign", type=Path)
    validate.add_argument("plan", type=Path)

    submit = commands.add_parser("submit", help="submit the exact reviewed plan")
    submit.add_argument("campaign", type=Path)
    submit.add_argument("plan", type=Path)

    seal = commands.add_parser("seal", help="irreversibly seal a Campaign roster")
    seal.add_argument("campaign", type=Path)

    status = commands.add_parser("status", help="show scheduler-only Campaign evidence as JSON")
    status.add_argument("campaign", type=Path)

    logs = commands.add_parser(
        "logs",
        help="write sensitive untrusted raw log bytes; redirect to a private file or safe viewer",
        description=(
            "Write sensitive untrusted raw log bytes, which may contain terminal control "
            "sequences. Redirect output to a private file or safe binary viewer."
        ),
    )
    logs.add_argument("campaign", type=Path)
    logs.add_argument("allocation_id")
    logs.add_argument("--task", metavar="TASK_KEY")
    logs.add_argument("--bytes", type=int, default=65_536, metavar="N")

    reconcile = commands.add_parser("reconcile", help="query one ambiguous allocation once")
    reconcile.add_argument("campaign", type=Path)
    reconcile.add_argument("allocation_id")

    resolve = commands.add_parser("resolve", help="record an operator ambiguity decision")
    resolve.add_argument("campaign", type=Path)
    resolve.add_argument("allocation_id")
    decision = resolve.add_mutually_exclusive_group(required=True)
    decision.add_argument("--job-id", type=int)
    decision.add_argument("--not-submitted", action="store_true")
    resolve.add_argument("--cluster")
    return parser


def _load_tasks(path: Path) -> tuple[Task, ...]:
    values: list[Task] = []
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise ServatusError(f"cannot read task file: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            raw = cast(object, json.loads(line))
            if not isinstance(raw, dict):
                raise ValueError("expected key, args, and stdin_file")
            mapping = cast(dict[str, object], raw)
            if set(mapping) != {"key", "args", "stdin_file"}:
                raise ValueError("expected key, args, and stdin_file")
            args = mapping["args"]
            if not isinstance(args, list):
                raise ValueError("args must be an array of strings")
            typed_args: list[str] = []
            for value in cast(list[object], args):
                if not isinstance(value, str):
                    raise ValueError("args must be an array of strings")
                typed_args.append(value)
            stdin_file = Path(cast(str, mapping["stdin_file"]))
            if not stdin_file.is_absolute():
                stdin_file = path.parent / stdin_file
            values.append(
                Task(cast(str, mapping["key"]), tuple(typed_args), stdin_file.read_bytes())
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ServatusError(f"invalid task file line {line_number}") from error
    return tuple(values)


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ServatusError(f"cannot read plan document: {path}") from error


def _write_json(path: Path, value: object) -> None:
    encoded = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)

    def write(stage: Path) -> None:
        stage.chmod(0o600)
        stage.write_bytes(encoded)

    publish_file(path, write)


def _receipt_json(receipt: JobReceipt) -> dict[str, object]:
    return {
        "allocation_id": receipt.allocation_id,
        "job_id": receipt.job_id,
        "cluster": receipt.cluster,
        "task_keys": list(receipt.task_keys),
    }


def _run(arguments: argparse.Namespace) -> None:
    command = cast(str, arguments.command)
    if command == "plan":
        campaign = Campaign.open(arguments.campaign, _load_tasks(arguments.tasks))
        profile = Profile.load(Path.cwd() / "SERVATUS.toml", name=arguments.profile)
        view = campaign.inspect()
        plan = campaign.plan(
            profile,
            view=view,
            retry=arguments.retry,
            allow_duplicate_risk=arguments.allow_duplicate_risk,
            tasks_per_allocation=arguments.tasks_per_allocation,
        )
        _write_json(arguments.output, plan_document(plan))
        for warning in plan.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        if arguments.show_scripts:
            print(
                "warning: complete scripts expose task arguments and payloads",
                file=sys.stderr,
            )
            print(json.dumps(sensitive_script_document(plan), sort_keys=True))
        else:
            print(plan.digest)
    elif command == "validate":
        campaign = Campaign.load(arguments.campaign)
        results = campaign.validate(restore_plan(campaign, _read_json(arguments.plan)))
        print(json.dumps(validation_document(results), sort_keys=True))
    elif command == "submit":
        campaign = Campaign.load(arguments.campaign)
        receipts = campaign.submit(restore_plan(campaign, _read_json(arguments.plan)))
        print(json.dumps([_receipt_json(receipt) for receipt in receipts], sort_keys=True))
    elif command == "seal":
        Campaign.load(arguments.campaign).seal()
        print(json.dumps({"sealed": True}, sort_keys=True))
    elif command == "status":
        view = Campaign.load(arguments.campaign).inspect()
        print(json.dumps(campaign_view_document(view), sort_keys=True))
    elif command == "logs":
        snapshot = Campaign.load(arguments.campaign).read_log(
            arguments.allocation_id,
            task_key=arguments.task,
            max_bytes=arguments.bytes,
        )
        sys.stdout.buffer.write(snapshot.content)
    elif command == "reconcile":
        campaign = Campaign.load(arguments.campaign)
        receipt = campaign.reconcile(arguments.allocation_id)
        print(json.dumps(_receipt_json(receipt), sort_keys=True))
    elif command == "resolve":
        campaign = Campaign.load(arguments.campaign)
        campaign.resolve(
            arguments.allocation_id,
            job_id=None if arguments.not_submitted else arguments.job_id,
            cluster=arguments.cluster,
        )
        print(json.dumps({"allocation_id": arguments.allocation_id, "resolved": True}))


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    try:
        arguments = parser.parse_args(argv)
        _run(arguments)
    except ServatusError as error:
        parser.error(str(error))
    return 0
