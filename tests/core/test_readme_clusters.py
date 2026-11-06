"""The README's cluster configurations stay valid, whatever else changes."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from servatus.campaign import Profile, Verdict, diagnose
from servatus.testing import FakeScheduler

_README = Path(__file__).parents[2] / "README.md"
_CLUSTERS = _README.read_text().partition("## Clusters")[2].partition("\n## ")[0]
_DOCUMENTS = re.findall(r"```toml\n(.*?)```", _CLUSTERS, re.DOTALL)
_PROFILES = {"university": {"rtx2080", "l40", "cpu"}}
_PROFILES["research"] = {"a100", "l40s", "l40", "h100", "cpu", "eval", "smoke", "l40-solo"}


def test_the_readme_configures_both_clusters() -> None:
    assert len(_DOCUMENTS) == 2


@pytest.mark.parametrize(
    ("document", "host"), [(_DOCUMENTS[0], "university"), (_DOCUMENTS[1], "research")]
)
def test_every_profile_in_the_readme_loads_and_passes_doctor(
    document: str, host: str, tmp_path: Path
) -> None:
    path = tmp_path / "SERVATUS.toml"
    path.write_text(document)
    for name in _PROFILES[host]:
        profile = Profile.load(path, name=name)
        assert profile.target.host == host
        assert diagnose(profile, connect=FakeScheduler()).verdict is Verdict.PASS
    assert re.findall(r"^\[profiles\.([a-z0-9-]+)\.target\]$", document, re.M).__len__() == len(
        _PROFILES[host]
    )
