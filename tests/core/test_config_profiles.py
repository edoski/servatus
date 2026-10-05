from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from servatus.campaign._config import Apptainer, Profile, Resources
from servatus.errors import ConfigurationError, NotFound

CPU = """\
[profiles.cpu.target]
host = "login.example.edu"
slurm_bin = "/opt/slurm/bin"
apptainer = "/usr/bin/apptainer"
image = "/images/cpu.sif"
work_root = "/work"
log_root = "/logs"
partitions = ["cpu"]
max_tasks_per_allocation = 4
max_cpus_per_allocation = 16
max_memory_mib_per_allocation = 8192
max_time_limit = "1-00:00:00"
[profiles.cpu.resources]
cpus = 2
memory_mib = 1024
time_limit = "00:10:00"
"""

SHARED = """\
default_profile = "cpu"
[target]
host = "login.example.edu"
slurm_bin = "/opt/slurm/bin"
apptainer = "/usr/bin/apptainer"
image = "/images/shared.sif"
binds = ["/data:/data:ro"]
work_root = "/work"
log_root = "/logs"
partitions = ["cpu"]
max_tasks_per_allocation = 4
max_cpus_per_allocation = 16
max_memory_mib_per_allocation = 8192
max_time_limit = "1-00:00:00"
[resources]
cpus = 2
memory_mib = 1024
time_limit = "00:10:00"
[profiles.cpu]
[profiles.gpu.target]
partitions = ["gpu"]
gpu_gres = "gpu:a100"
max_gpus_per_allocation = 4
[profiles.gpu.resources]
gpus = 1
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "SERVATUS.toml"
    path.write_text(text, encoding="utf-8")
    return path


def two(default: str | None = None) -> str:
    prefix = "" if default is None else f'default_profile = "{default}"\n'
    return prefix + CPU + CPU.replace("profiles.cpu", "profiles.alias")


def test_sole_profile_selects_itself_with_safe_defaults(tmp_path: Path) -> None:
    value = Profile.load(write(tmp_path, CPU))
    assert value.label == "cpu"
    assert value.resources == Resources(cpus=2, memory_mib=1024, time_limit="00:10:00")
    assert value.resources.gpus == 0 and value.resources.signal_before_end is None
    target = value.target
    assert target.container == Apptainer(executable="/usr/bin/apptainer", image="/images/cpu.sif")
    assert target.account is target.qos is target.constraint is target.gpu_gres is None
    assert target.max_allocations_per_submit is None
    assert target.max_script_bytes == 1024 * 1024
    assert target.max_time_limit == timedelta(days=1)


def test_profile_accepts_string_paths(tmp_path: Path) -> None:
    assert Profile.load(str(write(tmp_path, CPU))).label == "cpu"


def test_host_and_container_are_optional(tmp_path: Path) -> None:
    text = (
        CPU.replace('host = "login.example.edu"\n', "")
        .replace('apptainer = "/usr/bin/apptainer"\n', "")
        .replace('image = "/images/cpu.sif"\n', "")
    )
    value = Profile.load(write(tmp_path, text))
    assert value.target.host is None
    assert value.target.container is None


def test_profiles_override_document_defaults_per_key(tmp_path: Path) -> None:
    path = write(tmp_path, SHARED)
    cpu = Profile.load(path)
    gpu = Profile.load(path, name="gpu")
    assert (cpu.label, gpu.label) == ("cpu", "gpu")
    assert cpu.target.partitions == ("cpu",) and cpu.target.gpu_gres is None
    assert gpu.target.partitions == ("gpu",) and gpu.target.gpu_gres == "gpu:a100"
    assert cpu.target.container is not None
    assert cpu.target.container.binds == ("/data:/data:ro",)
    assert replace(gpu.target, partitions=("cpu",), gpu_gres=None, max_gpus_per_allocation=0) == (
        cpu.target
    )
    assert gpu.resources == Resources(cpus=2, memory_mib=1024, time_limit="00:10:00", gpus=1)


def test_container_keys_override_individually(tmp_path: Path) -> None:
    text = SHARED + '[profiles.other.target]\nimage = "/images/other.sif"\n'
    container = Profile.load(write(tmp_path, text), name="other").target.container
    assert container == Apptainer(
        executable="/usr/bin/apptainer", image="/images/other.sif", binds=("/data:/data:ro",)
    )


def test_explicit_name_overrides_default_and_aliases_keep_values(tmp_path: Path) -> None:
    path = write(tmp_path, two(default="cpu"))
    assert Profile.load(path).label == "cpu"
    alias = Profile.load(path, name="alias")
    assert alias.label == "alias"
    assert (alias.target, alias.resources) == (
        Profile.load(path).target,
        Profile.load(path).resources,
    )


@pytest.mark.parametrize("label", ["cpu lane", "計算 🚀", "cpu\tlane"])
def test_profile_labels_are_opaque_toml_keys(tmp_path: Path, label: str) -> None:
    encoded = json.dumps(label, ensure_ascii=False)
    path = write(tmp_path, CPU.replace("profiles.cpu", f"profiles.{encoded}"))
    assert Profile.load(path).label == label
    assert Profile.load(path, name=label).label == label


def test_unselected_profiles_are_not_validated_semantically(tmp_path: Path) -> None:
    text = two(default="cpu").replace(
        '[profiles.alias.target]\nhost = "login.example.edu"',
        '[profiles.alias.target]\nhost = "invalid host"',
    )
    path = write(tmp_path, text)
    assert Profile.load(path).label == "cpu"
    with pytest.raises(ConfigurationError, match="SSH destination"):
        Profile.load(path, name="alias")


def test_overridden_document_defaults_are_not_validated(tmp_path: Path) -> None:
    text = (
        SHARED.replace('host = "login.example.edu"', 'host = "invalid host"')
        + '[profiles.fixed.target]\nhost = "login.example.edu"\n'
    )
    path = write(tmp_path, text)
    assert Profile.load(path, name="fixed").target.host == "login.example.edu"
    with pytest.raises(ConfigurationError, match="SSH destination"):
        Profile.load(path)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "profiles must be a nonempty table"),
        ("profiles = {}\n", "profiles must be a nonempty table"),
        ("profiles = 1\n", "profiles must be a nonempty table"),
        ("profiles.cpu = 1\n", "profile 'cpu' must be a table"),
        (two(), "profile selection is required"),
        (two(default="missing"), "default_profile must name a declared profile"),
        ("default_profile = 3\n" + CPU, "default_profile must name a declared profile"),
        ("extra = true\n" + CPU, "unknown document keys: extra"),
        (
            CPU.replace("[profiles.cpu.resources]", "[profiles.cpu.extra]"),
            "unknown keys in profile",
        ),
        (CPU + "misspelled = 1\n", "unknown resources keys in profile 'cpu': misspelled"),
        (
            CPU.replace("[profiles.cpu.target]", "[profiles.cpu.target]\ncontainer = 1"),
            "unknown target keys in profile 'cpu': container",
        ),
        (CPU.replace("cpus = 2\n", ""), "missing resources keys: cpus"),
        (CPU.replace('slurm_bin = "/opt/slurm/bin"\n', ""), "missing target keys: slurm_bin"),
        (CPU.replace('apptainer = "/usr/bin/apptainer"\n', ""), "missing target keys: apptainer"),
        (CPU.replace('image = "/images/cpu.sif"\n', ""), "missing target keys: image"),
        (CPU.replace('partitions = ["cpu"]', 'partitions = "cpu"'), "array of strings"),
        (CPU.replace('"00:10:00"', "00:10:00"), "duration string or timedelta"),
        (CPU.replace("cpus = 2", "cpus = 2.0"), "cpus must be an integer"),
        (CPU.replace("profiles.cpu", 'profiles.""'), "profile label must be nonempty"),
        (SHARED.replace("[target]", "[target]\nmisspelled = 1"), "unknown target keys in the"),
        (SHARED.replace("[resources]", "[resources]\nmisspelled = 1"), "unknown resources keys"),
        ("target = 1\n" + CPU, "target in the document must be a table"),
        (SHARED.replace("[target]\n", "target = 1\n[unused]\n"), "unknown document keys: unused"),
        (
            two(default="cpu").replace("[profiles.alias.target]", "[profiles.alias.target]\nx = 1"),
            "unknown target keys in profile 'alias': x",
        ),
        ("[[profiles]]\n", "profiles must be a nonempty table"),
        ("not toml", "cannot read TOML configuration"),
    ],
)
def test_malformed_documents_are_rejected(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        Profile.load(write(tmp_path, text))


def test_explicit_undeclared_name_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="profile 'missing' is not declared"):
        Profile.load(write(tmp_path, CPU), name="missing")


def test_missing_document_is_not_found(tmp_path: Path) -> None:
    with pytest.raises(NotFound, match="configuration file does not exist"):
        Profile.load(tmp_path / "SERVATUS.toml")


def test_unreadable_documents_are_configuration_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="cannot read TOML configuration"):
        Profile.load(tmp_path)
    path = tmp_path / "SERVATUS.toml"
    path.write_bytes(b'[profiles.cpu]\nlabel = "\xff"\n')
    with pytest.raises(ConfigurationError, match="cannot read TOML configuration"):
        Profile.load(path)
