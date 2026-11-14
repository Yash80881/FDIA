"""Shared pytest fixtures: every test uses a temporary database / output folder."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import load_settings  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    c = load_settings(overrides={
        "paths.database": str(tmp_path / "test.db"),
        "paths.exports": str(tmp_path / "exports"),
        "paths.processed": str(tmp_path / "processed"),
        "paths.raw": str(tmp_path / "raw"),
        "paths.logs": str(tmp_path / "logs"),
        "paths.results": str(tmp_path / "results"),
        "paths.stop_flag": str(tmp_path / "STOP"),
        "paths.models": str(tmp_path / "models"),
    })
    return c


@pytest.fixture(scope="session")
def grid():
    from simulator.grid_model import GridModel
    return GridModel()
