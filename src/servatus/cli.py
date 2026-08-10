# pyright: reportPrivateUsage=false

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import suppress
from pathlib import Path
from typing import cast

from ._campaign import (
    Campaign,
    JobReceipt,
    ResourceRequest,
    SlurmTarget,
    Task,
    plan_document,
    restore_plan,
    sensitive_script_document,
    validate_plan,
    validation_document,
)
from ._errors import ServatusError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="servatus", description="Durable Slurm work campaigns")
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="build a fully local immutable plan")
    plan.add_argument("tasks", type=Path)
    plan.add_argument("--target", type=Path, required=True)
    plan.add_argument("--resources", type=Path, required=True)
    plan.add_argument("--campaign", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--tasks-per-allocation", type=int)
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

    status = commands.add_parser("status", help="show local campaign provenance")
    status.add_argument("campaign", type=Path)

    reconcile = commands.add_parser("reconcile", help="query one ambiguous allocation once")
    reconcile.add_argument("campaign", type=Path)
    reconcile.add_argument("allocation_id")
    reconcile.add_argument("--target", type=Path, required=True)

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
    temporary = path.with_name(f".{path.name}-{os.urandom(8).hex()}.tmp")
    try:
        with temporary.open("xb") as destination:
            os.chmod(temporary, 0o600)
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


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
        plan = campaign.plan(
            SlurmTarget.from_toml(arguments.target),
            ResourceRequest.from_toml(arguments.resources),
            tasks_per_allocation=arguments.tasks_per_allocation,
        )
        _write_json(arguments.output, plan_document(plan))
        if arguments.show_scripts:
            print(
                "warning: complete scripts expose task arguments and payloads",
                file=sys.stderr,
            )
            print(json.dumps(sensitive_script_document(plan), sort_keys=True))
        else:
            print(plan.digest)
    elif command == "validate":
        campaign = Campaign._reopen(arguments.campaign)
        results = validate_plan(restore_plan(campaign, _read_json(arguments.plan)))
        print(json.dumps(validation_document(results), sort_keys=True))
    elif command == "submit":
        campaign = Campaign._reopen(arguments.campaign)
        receipts = campaign.submit(restore_plan(campaign, _read_json(arguments.plan)))
        print(json.dumps([_receipt_json(receipt) for receipt in receipts], sort_keys=True))
    elif command == "status":
        status = Campaign._reopen(arguments.campaign).status()
        print(
            json.dumps(
                {
                    "pending_task_keys": list(status.pending_task_keys),
                    "receipts": [_receipt_json(receipt) for receipt in status.receipts],
                    "ambiguous_allocation_ids": list(status.ambiguous_allocation_ids),
                },
                sort_keys=True,
            )
        )
    elif command == "reconcile":
        campaign = Campaign._reopen(arguments.campaign)
        receipt = campaign.reconcile(
            SlurmTarget.from_toml(arguments.target), arguments.allocation_id
        )
        print(json.dumps(_receipt_json(receipt), sort_keys=True))
    elif command == "resolve":
        campaign = Campaign._reopen(arguments.campaign)
        campaign.resolve(
            arguments.allocation_id,
            job_id=None if arguments.not_submitted else arguments.job_id,
            cluster=arguments.cluster,
        )
        print(json.dumps({"allocation_id": arguments.allocation_id, "resolved": True}))
    else:
        raise AssertionError(f"unhandled command: {command}")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    try:
        arguments = parser.parse_args(argv)
        _run(arguments)
    except ServatusError as error:
        parser.error(str(error))
    return 0
