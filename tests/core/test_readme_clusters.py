"""The README's cluster configurations stay valid, whatever else changes."""

from __future__ import annotations

import re
from pathlib import Path

from servatus.campaign import Profile, Verdict, diagnose
from servatus.testing import FakeScheduler

_README = Path(__file__).parents[2] / "README.md"
_PROFILES = {
    "university": {"rtx2080", "l40", "cpu"},
    "research": {"a100", "l40s", "l40", "h100", "cpu", "eval", "smoke", "l40-solo"},
}


def test_every_profile_in_the_readme_loads_and_passes_doctor(tmp_path: Path) -> None:
    clusters = _README.read_text().partition("## Clusters")[2].partition("\n## ")[0]
    documents = re.findall(r"```toml\n(.*?)```", clusters, re.DOTALL)
    hosts = [re.search(r'^host = "(.*)"$', document, re.M) for document in documents]
    assert [host.group(1) for host in hosts if host] == list(_PROFILES)
    path = tmp_path / "SERVATUS.toml"
    for document, (host, names) in zip(documents, _PROFILES.items(), strict=True):
        assert set(re.findall(r"^\[profiles\.([a-z0-9-]+)\.target\]$", document, re.M)) == names
        path.write_text(document)
        for name in names:
            profile = Profile.load(path, name=name)
            assert profile.target.host == host
            assert diagnose(profile, connect=FakeScheduler()).verdict is Verdict.PASS
