"""Shared builders for synthetic Campaign inputs. Never point these at a real cluster."""

from __future__ import annotations

from typing import Any

from servatus.campaign._config import Apptainer, Profile, Resources, Target, Task


def target(**changes: Any) -> Target:
    values: dict[str, Any] = {
        "host": "login.example.edu",
        "slurm_bin": "/opt/slurm/bin",
        "work_root": "/cluster/work/project",
        "log_root": "/cluster/logs/project",
        "partitions": ("gpu",),
        "container": Apptainer(executable="/usr/bin/apptainer", image="/cluster/images/work.sif"),
        "account": "research",
        "gpu_gres": "gpu:a100",
        "max_tasks_per_allocation": 4,
        "max_cpus_per_allocation": 128,
        "max_memory_mib_per_allocation": 262144,
        "max_gpus_per_allocation": 4,
        "max_time_limit": "7-00:00:00",
    }
    values.update(changes)
    return Target(**values)


def resources(**changes: Any) -> Resources:
    values: dict[str, Any] = {
        "cpus": 32,
        "memory_mib": 65536,
        "gpus": 1,
        "time_limit": "3-00:00:00",
    }
    values.update(changes)
    return Resources(**values)


def profile(
    target_value: Target | None = None,
    resource_value: Resources | None = None,
    *,
    label: str = "test",
) -> Profile:
    return Profile(label, target_value or target(), resource_value or resources())


def tasks(count: int, *, prefix: str = "task") -> tuple[Task, ...]:
    return tuple(
        Task(f"{prefix}-{index}", ("train", "--index", str(index)), stdin=f"{index}\n".encode())
        for index in range(count)
    )
