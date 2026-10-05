from __future__ import annotations

import copy
import pickle
from datetime import timedelta
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
from support.builders import profile, resources, target

from servatus.campaign import _codec
from servatus.campaign._config import Apptainer, Profile, Resources, Target, Task
from servatus.errors import ConfigurationError

# --- Task ------------------------------------------------------------------------------------


def test_task_freezes_arguments_and_defaults_to_empty_stdin_and_env() -> None:
    args = ["worker", "--flag"]
    task = Task("task", args)
    args.append("changed")
    assert task.args == ("worker", "--flag")
    assert task.stdin == b""
    assert dict(task.env) == {}
    assert Task("task") == Task("task", ())


def test_task_freezes_environment_sorted_by_name() -> None:
    env = {"ZED": "1", "ALPHA": "a,b c", "EMPTY": ""}
    task = Task("task", (), env=env)
    env["ADDED"] = "later"
    assert list(task.env.items()) == [("ALPHA", "a,b c"), ("EMPTY", ""), ("ZED", "1")]
    assert task == Task("task", env={"ZED": "1", "EMPTY": "", "ALPHA": "a,b c"})
    assert task != Task("task")
    with pytest.raises(TypeError, match="does not support item assignment"):
        task.env["ADDED"] = "later"  # pyright: ignore[reportIndexIssue]


def test_task_is_immutable() -> None:
    task = Task("task")
    with pytest.raises(AttributeError, match="key"):
        task.key = "other"  # pyright: ignore[reportAttributeAccessIssue]


def test_task_pickles_deep_copies_and_hashes_by_value() -> None:
    task = Task("k/ü", ["a", ""], stdin=b"\0\xff", env={"B": "1", "A": "é"})
    assert pickle.loads(pickle.dumps(task)) == task
    assert copy.deepcopy(task) == task
    assert copy.copy(task) == task
    assert hash(task) == hash(Task("k/ü", ("a", ""), stdin=b"\0\xff", env={"A": "é", "B": "1"}))
    assert hash(pickle.loads(pickle.dumps(task))) == hash(task)
    assert len({task, copy.deepcopy(task)}) == 1


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("", "must be nonempty"),
        (7, "must be a string"),
        ("a\0b", "cannot contain NUL"),
        ("a\udcffb", "must be valid UTF-8"),
    ],
)
def test_task_key_must_be_nonempty_utf8_text(key: object, message: str) -> None:
    with pytest.raises(ConfigurationError, match=f"Task.key {message}"):
        Task(key)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("args", ["worker", b"worker", bytearray(), {"a": "b"}, 3])
def test_task_rejects_non_sequence_arguments(args: object) -> None:
    with pytest.raises(ConfigurationError, match="sequence of strings"):
        Task("task", args)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["a", 2], "must be a string"),
        (["a\0"], "cannot contain NUL"),
        (["\udcff"], "must be valid UTF-8"),
    ],
)
def test_task_rejects_invalid_argument_text(args: list[object], message: str) -> None:
    with pytest.raises(ConfigurationError, match=f"Task.args {message}"):
        Task("task", args)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("stdin", ["text", bytearray(b"x"), None])
def test_task_stdin_must_be_bytes(stdin: object) -> None:
    with pytest.raises(ConfigurationError, match="stdin must be bytes"):
        Task("task", stdin=stdin)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ([("A", "1")], "map environment names"),
        ("A=1", "map environment names"),
        ({"1A": "x"}, "identifiers"),
        ({"A-B": "x"}, "identifiers"),
        ({"": "x"}, "identifiers"),
        ({1: "x"}, "identifiers"),
        ({"A": 1}, r"Task.env\[A\] must be a string"),
        ({"A": "x\0"}, "NUL"),
        ({"A": "\udcff"}, "valid UTF-8"),
        ({"SERVATUS_X": "1"}, "cannot start with SERVATUS_"),
    ],
)
def test_task_rejects_invalid_environment(env: object, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        Task("task", env=env)  # pyright: ignore[reportArgumentType]


# --- Resources -------------------------------------------------------------------------------


def test_resources_round_time_limit_up_to_whole_minutes() -> None:
    value = Resources(cpus=1, memory_mib=1, time_limit="01:00:01")
    assert value.time_limit == timedelta(hours=1, minutes=1)
    assert value.gpus == 0 and value.signal_before_end is None
    assert Resources(cpus=1, memory_mib=1, time_limit=timedelta(seconds=59)).time_limit == (
        timedelta(minutes=1)
    )
    signal = Resources(cpus=1, memory_mib=1, time_limit="00:10:00", signal_before_end="00:01:30")
    assert signal.signal_before_end == timedelta(seconds=90)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"cpus": True}, "cpus must be an integer >= 1"),
        ({"cpus": 0}, "cpus must be an integer >= 1"),
        ({"cpus": 1.0}, "cpus must be an integer >= 1"),
        ({"memory_mib": 0}, "memory_mib must be an integer >= 1"),
        ({"gpus": -1}, "gpus must be an integer >= 0"),
        ({"time_limit": "00:00:00"}, "positive, not unlimited"),
        ({"time_limit": "UNLIMITED"}, "days-"),
        ({"time_limit": 60}, "duration string or timedelta"),
        ({"time_limit": timedelta(milliseconds=1500)}, "whole seconds"),
        ({"signal_before_end": "3-00:00:00"}, "shorter than time_limit"),
        ({"time_limit": "2-00:00:00", "signal_before_end": "1-00:00:00"}, "Slurm's bound"),
        ({"signal_before_end": "00:00:00"}, "positive"),
    ],
)
def test_resources_reject_invalid_values(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        resources(**changes)


def test_resources_are_keyword_only() -> None:
    with pytest.raises(TypeError, match="positional"):
        Resources(1, 1, "01:00:00")  # pyright: ignore[reportCallIssue]


# --- Target and Apptainer --------------------------------------------------------------------


def test_target_defaults_are_conservative_and_optional() -> None:
    value = Target(
        slurm_bin="/opt/slurm/bin",
        work_root="/work",
        log_root="/logs",
        partitions=["cpu"],
        max_tasks_per_allocation=1,
        max_cpus_per_allocation=1,
        max_memory_mib_per_allocation=1,
        max_time_limit="00:30:01",
    )
    assert value.host is None and value.container is None
    assert value.account is value.qos is value.constraint is value.gpu_gres is None
    assert value.max_gpus_per_allocation == 0
    assert value.max_allocations_per_submit is None
    assert value.max_script_bytes == 1024 * 1024
    assert value.partitions == ("cpu",)
    assert value.max_time_limit == timedelta(minutes=31)


def test_target_freezes_partitions_and_normalizes_paths() -> None:
    partitions = ["cpu", "gpu"]
    value = target(partitions=partitions, slurm_bin=Path("/opt//slurm/./bin"))
    partitions.append("changed")
    assert value.partitions == ("cpu", "gpu")
    assert value.slurm_bin == PurePosixPath("/opt/slurm/bin")
    assert type(value.slurm_bin) is PurePosixPath


@pytest.mark.parametrize("host", [None, "login", "user@login.example.edu", "alias_1", "a-b.c"])
def test_target_accepts_ssh_destinations(host: str | None) -> None:
    assert target(host=host).host == host


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"host": "-oProxyCommand=bad"}, "SSH destination"),
        ({"host": "two words"}, "SSH destination"),
        ({"host": ""}, "SSH destination"),
        ({"slurm_bin": "relative/bin"}, "slurm_bin must be an absolute POSIX path"),
        ({"work_root": "/a/../b"}, "work_root must be an absolute POSIX path"),
        ({"log_root": "/logs\n"}, "log_root must be an absolute POSIX path"),
        ({"log_root": 7}, "log_root must be an absolute POSIX path"),
        ({"work_root": "/w\udcff"}, "work_root must be valid UTF-8"),
        ({"work_root": "/w,x"}, "work_root cannot contain ',': it is bound into containers"),
        ({"work_root": "/w:x"}, "work_root cannot contain ':': it is bound"),
        ({"work_root": "/a:b,c"}, "work_root cannot contain ',' or ':'"),
        ({"log_root": "/logs/%u"}, "log_root cannot contain '%': Slurm expands"),
        ({"partitions": ()}, "partitions must be nonempty"),
        ({"partitions": ("a", "a")}, "partitions must be unique"),
        ({"partitions": "gpu"}, "partitions must be a sequence"),
        ({"partitions": ("bad name",)}, "partition must be one safe site token"),
        ({"account": "-x"}, "account must be one safe site token"),
        ({"qos": "a b"}, "qos must be one safe site token"),
        ({"constraint": "a|b"}, "constraint must be one safe site token"),
        ({"gpu_gres": "scratch"}, "count-free GPU resource"),
        ({"gpu_gres": "gpu:2"}, "count-free GPU resource"),
        ({"gpu_gres": "gpu:a100:2"}, "count-free GPU resource"),
        ({"gpu_gres": None}, "gpu_gres and max_gpus_per_allocation conflict"),
        ({"max_gpus_per_allocation": 0}, "gpu_gres and max_gpus_per_allocation conflict"),
        ({"max_tasks_per_allocation": 0}, "max_tasks_per_allocation must be an integer >= 1"),
        ({"max_cpus_per_allocation": False}, "max_cpus_per_allocation must be an integer"),
        ({"max_memory_mib_per_allocation": 0}, "max_memory_mib_per_allocation must be"),
        ({"max_time_limit": "0:00:00"}, "max_time_limit"),
        ({"max_allocations_per_submit": 0}, "max_allocations_per_submit must be an integer"),
        ({"max_script_bytes": 0}, "max_script_bytes must be an integer >= 1"),
        ({"container": "apptainer"}, "container is invalid"),
    ],
)
def test_target_rejects_invalid_values(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        target(**changes)


@pytest.mark.parametrize("gres", ["gpu", "gpu:a100", "gpu:h100.80gb", "gpu:A_1"])
def test_target_accepts_count_free_gpu_gres(gres: str) -> None:
    assert target(gpu_gres=gres).gpu_gres == gres


@pytest.mark.parametrize(
    "binds", [(), ("/data",), ("/data:/mnt",), ("/data:/mnt:ro",), ("/a:/b:rw", "/c")]
)
def test_apptainer_accepts_bind_forms(binds: tuple[str, ...]) -> None:
    value = Apptainer(executable="/usr/bin/apptainer", image="/images/a.sif", binds=list(binds))
    assert value.binds == binds


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"executable": "apptainer"}, "apptainer must be an absolute POSIX path"),
        ({"image": "/images/../a.sif"}, "image must be an absolute POSIX path"),
        ({"image": "/images/docker:a.sif"}, "image cannot contain ':': Apptainer reads it"),
        ({"binds": "/data"}, "binds must be a sequence"),
        ({"binds": ("data",)}, "bind path must be an absolute"),
        ({"binds": ("/a,/b",)}, "commas"),
        ({"binds": ("/a:/b:rx",)}, "ro or rw"),
        ({"binds": ("/a:/b:ro:x",)}, "SOURCE"),
        ({"binds": ("",)}, "bind must be nonempty"),
    ],
)
def test_apptainer_rejects_invalid_values(changes: dict[str, Any], message: str) -> None:
    values: dict[str, Any] = {"executable": "/usr/bin/apptainer", "image": "/images/a.sif"}
    values.update(changes)
    with pytest.raises(ConfigurationError, match=message):
        Apptainer(**values)


# --- Profile ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "message"),
    [("", "must be nonempty"), (7, "must be a string"), ("a\0", "cannot contain NUL")],
)
def test_profile_label_is_nonempty_text(label: object, message: str) -> None:
    with pytest.raises(ConfigurationError, match=f"profile label {message}"):
        Profile(label, target(), resources())  # pyright: ignore[reportArgumentType]


def test_profile_requires_typed_parts() -> None:
    with pytest.raises(ConfigurationError, match="profile target must be a Target"):
        Profile("p", "target", resources())  # pyright: ignore[reportArgumentType]
    with pytest.raises(ConfigurationError, match="profile resources must be Resources"):
        Profile("p", target(), {})  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("label", ["cpu lane", "計算 🚀", "cpu\tlane"])
def test_profile_labels_are_opaque(label: str) -> None:
    assert Profile(label, target(), resources()).label == label


@pytest.mark.parametrize(
    "value",
    [
        profile(),
        profile(
            target(container=None, host=None, gpu_gres=None, max_gpus_per_allocation=0),
            resources(gpus=0, signal_before_end="00:05:00"),
            label="ü",
        ),
        profile(target(container=Apptainer(executable="/a", image="/b", binds=("/c:/d:ro",)))),
    ],
)
def test_profiles_round_trip_through_the_codec(value: Profile) -> None:
    document = _codec.decode_json(_codec.canonical(_codec.dump(value)))
    assert _codec.load(Profile, document) == value
    assert pickle.loads(pickle.dumps(value)) == value
