"""AGENTS.md and CLAUDE.md stay present, linked and true to the tree."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Backticked text that looks like a path but is not one in a clean checkout.
NOT_PATHS = frozenset(
    {"aurora-state-171-effective/v4", "aurora-state-171/v1", "data/", "runs/"}
)

PATH_SUFFIXES = (
    ".py",
    ".md",
    ".toml",
    ".yml",
    ".yaml",
    ".txt",
    ".json",
    ".cff",
    ".js",
    ".svg",
)
PATH_TOKEN = re.compile(r"[A-Za-z0-9_.\-/]+")


def _inline_code(text: str) -> list[str]:
    without_fences = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return re.findall(r"`([^`\n]+)`", without_fences)


def _named_paths(text: str) -> set[str]:
    return {
        token
        for token in _inline_code(text)
        if PATH_TOKEN.fullmatch(token)
        and token not in NOT_PATHS
        and ("/" in token or token.endswith(PATH_SUFFIXES))
    }


def test_agents_file_names_only_paths_that_exist():
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    named = _named_paths(text)
    assert len(named) >= 10, "the path scan found too little; check the parser"
    missing = sorted(path for path in named if not (ROOT / path).exists())
    assert not missing, f"AGENTS.md names paths that do not exist: {missing}"


def test_claude_file_imports_the_agents_file():
    lines = (ROOT / "CLAUDE.md").read_text(encoding="utf-8").splitlines()
    assert "@AGENTS.md" in lines, "CLAUDE.md must import AGENTS.md on its own line"


def test_agents_file_keeps_its_binding_sections():
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    for heading in (
        "## 1. Hard stops",
        "## 2. No workarounds",
        "## 8. Protected files",
        "## 9. Automated and self-improving pipelines",
        "## 10. What is enforced and what is on trust",
    ):
        assert heading in text, f"AGENTS.md lost the section {heading!r}"
