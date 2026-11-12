"""Shared pytest configuration for AuroraLM.

Keeps every test deterministic regardless of the order tests run in, and
offers the fixtures that would otherwise be rebuilt per module.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent

# Node ids of tests that take several seconds on CPU. Listed here because
# tests/test_smoke.py must stay runnable as a plain script without pytest.
SLOW_TESTS = frozenset(
    {
        "tests/test_smoke.py::test_gradient_flow",
    }
)


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.nodeid in SLOW_TESTS:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def _pinned_seed():
    """Start each test from the same RNG state; tests may reseed."""
    torch.manual_seed(0)
    yield


@pytest.fixture(scope="session")
def project_root() -> Path:
    """The repository root."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def device() -> torch.device:
    """CUDA when present, otherwise CPU (CI is CPU-only)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    """A fresh directory for datasets, runs and checkpoints a test writes."""
    path = tmp_path / "work"
    path.mkdir()
    return path
