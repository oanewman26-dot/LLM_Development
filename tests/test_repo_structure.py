"""Repository structure contract shared with the Aurora repository.

Asserts that the files the shared repository template requires exist, that
sources and tests stay in ``src/auroralm`` and ``tests``, and that nothing
private, generated or machine-local is tracked by git.
"""

from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "CONTRIBUTING.md",
    "AGENTS.md",
    "CLAUDE.md",
    "SECURITY.md",
    "CITATION.cff",
    ".gitignore",
    ".gitattributes",
    ".github/workflows/ci.yml",
    ".github/dependabot.yml",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/change_request.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    "scripts/check_commit.py",
    "pyproject.toml",
    ".pre-commit-config.yaml",
    "conftest.py",
    "requirements.txt",
    "src/auroralm/__init__.py",
)

REQUIRED_DIRS = ("src/auroralm", "tests", "docs")

ROOT_PYTHON_FILES = frozenset({"conftest.py"})

# A tracked path may not contain any of these directory names.
FORBIDDEN_DIRS = frozenset(
    {
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".vscode",
        ".idea",
        ".obsidian",
        ".claude",
        "htmlcov",
        "build",
        "dist",
        "data",
        "datasets",
        "runs",
        "checkpoints",
        "wandb",
    }
)

FORBIDDEN_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".gguf",
    ".bin",
    ".odt",
    ".ods",
    ".tmp",
    ".log",
    ".bak",
)

FORBIDDEN_NAMES = frozenset(
    {
        ".env",
        ".coverage",
        "coverage.xml",
        ".DS_Store",
        "Thumbs.db",
        "workspace.json",
    }
)


def _tracked_files() -> list[str]:
    """Paths tracked by git, or skip when this is not a git checkout."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            capture_output=True,
            timeout=60,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        pytest.skip("not a git checkout, or git is unavailable")
    paths = [p for p in result.stdout.decode("utf-8").split("\0") if p]
    if not paths:
        pytest.skip("git reports no tracked files")
    return paths


def _violation(path: str) -> str | None:
    pure = PurePosixPath(path)
    bad_dirs = FORBIDDEN_DIRS.intersection(pure.parts[:-1])
    if bad_dirs:
        return f"inside forbidden directory {sorted(bad_dirs)[0]}/"
    if pure.name in FORBIDDEN_NAMES:
        return "secret, coverage output or editor state"
    if path.lower().endswith(FORBIDDEN_SUFFIXES):
        return "generated, binary or model artefact"
    return None


@pytest.mark.parametrize("relative", REQUIRED_FILES)
def test_required_file_exists(relative):
    assert (ROOT / relative).is_file(), f"missing required file: {relative}"


@pytest.mark.parametrize("relative", REQUIRED_DIRS)
def test_required_directory_exists(relative):
    assert (ROOT / relative).is_dir(), f"missing required directory: {relative}/"


def test_tests_are_collected_from_the_tests_directory():
    assert sorted((ROOT / "tests").glob("test_*.py")), "tests/ holds no test modules"
    stray = sorted(p.name for p in ROOT.glob("test_*.py"))
    assert not stray, f"test modules in the project root: {stray}"


def test_root_python_files_are_only_conftest():
    tracked_root = {p for p in _tracked_files() if "/" not in p and p.endswith(".py")}
    assert tracked_root - ROOT_PYTHON_FILES == set(), (
        "modules belong in src/auroralm/, tests in tests/"
    )


def test_no_forbidden_files_are_tracked():
    violations = {
        path: reason
        for path in _tracked_files()
        if (reason := _violation(path)) is not None
    }
    assert not violations, "forbidden files are tracked:\n" + "\n".join(
        f"  {path}: {reason}" for path, reason in sorted(violations.items())
    )


@pytest.mark.parametrize(
    "path",
    [
        ".venv/Lib/site-packages/x.py",
        "src/auroralm/__pycache__/NN.cpython-312.pyc",
        "runs/pilot/checkpoint.pt",
        "data/journal.jsonl",
        "weights/model.safetensors",
        ".env",
        "notes/draft.odt",
    ],
)
def test_forbidden_rules_reject_known_bad_paths(path):
    assert _violation(path) is not None


@pytest.mark.parametrize(
    "path",
    [
        "src/auroralm/NN.py",
        "artifacts/aurora_bpe_8192_v1/tokenizer.json",
        "calibration/activation_calibration_v2.json",
        "docs/Dev notes.txt",
    ],
)
def test_forbidden_rules_accept_versioned_paths(path):
    assert _violation(path) is None
