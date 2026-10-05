from __future__ import annotations

from pathlib import Path

import pytest
from campaign_world import World, make_world


@pytest.fixture
def world(tmp_path: Path) -> World:
    return make_world(tmp_path)
