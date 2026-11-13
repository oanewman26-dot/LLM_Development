"""Commit and branch policy checks (shared with the Aurora repository).

Standard library only, so the same file runs as a git hook and in CI before
any dependency is installed. Three subcommands:

    message <file>   validate the commit message git is about to record
                     (commit-msg hook)
    branch           refuse a commit made directly on a protected branch
                     (pre-commit hook)
    range <revs>     validate the subject of every non-merge commit in a
                     revision range such as origin/main..HEAD (CI)

A subject follows Conventional Commits: ``type(optional-scope): summary``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Sequence

TYPES = (
    "feat",
    "fix",
    "docs",
    "test",
    "refactor",
    "style",
    "perf",
    "build",
    "ci",
    "chore",
    "revert",
)
MAX_SUBJECT_LENGTH = 72
PROTECTED_BRANCHES = frozenset({"main"})

SUBJECT_PATTERN = re.compile(
    r"^(?P<type>[a-z]+)(\((?P<scope>[a-z0-9][a-z0-9._/-]*)\))?(?P<breaking>!)?: "
    r"(?P<summary>\S.*)$"
)
# Subjects git writes itself; they are accepted as they are.
GENERATED_PREFIXES = ("Merge ", 'Revert "', "fixup! ", "squash! ", "amend! ")
SCISSORS = "# ------------------------ >8 ------------------------"


def subject_problems(subject: str) -> list[str]:
    """Return what is wrong with a commit subject; empty when it conforms."""
    if subject.startswith(GENERATED_PREFIXES):
        return []
    if not subject.strip():
        return ["the subject line is empty"]
    problems: list[str] = []
    match = SUBJECT_PATTERN.match(subject)
    if match is None:
        problems.append(
            "the subject must read 'type(optional-scope): summary', "
            "with a lowercase type, a colon and one space"
        )
    else:
        if match["type"] not in TYPES:
            problems.append(
                f"unknown type '{match['type']}'; use one of {', '.join(TYPES)}"
            )
        if match["summary"].endswith("."):
            problems.append("the summary must not end with a full stop")
    if len(subject) > MAX_SUBJECT_LENGTH:
        problems.append(
            f"the subject is {len(subject)} characters; the limit is "
            f"{MAX_SUBJECT_LENGTH} (put the detail in the body)"
        )
    return problems


def message_problems(text: str) -> list[str]:
    """Return what is wrong with a whole commit message as git will store it."""
    lines: list[str] = []
    for line in text.splitlines():
        if line.rstrip() == SCISSORS:
            break
        if not line.startswith("#"):
            lines.append(line.rstrip())
    while lines and not lines[0]:
        lines.pop(0)
    if not lines:
        return ["the commit message is empty"]
    problems = subject_problems(lines[0])
    if len(lines) > 1 and lines[1]:
        problems.append("leave the second line blank, between subject and body")
    return problems


def current_branch() -> str | None:
    """Name of the checked-out branch, or None on a detached HEAD."""
    result = subprocess.run(
        ["git", "symbolic-ref", "--short", "-q", "HEAD"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return result.stdout.strip() or None


def range_subjects(revisions: str) -> list[tuple[str, str]]:
    """(short hash, subject) of each non-merge commit in a revision range."""
    result = subprocess.run(
        ["git", "log", "--no-merges", "--format=%h%x00%s", revisions, "--"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    pairs: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        commit, _, subject = line.partition("\0")
        pairs.append((commit, subject))
    return pairs


def _report(lines: Sequence[str]) -> None:
    sys.stderr.write("\n".join(lines) + "\n")


def _check_message(path: str) -> int:
    with open(path, encoding="utf-8", errors="replace") as handle:
        problems = message_problems(handle.read())
    if not problems:
        return 0
    _report(
        [
            "Commit message rejected (see CONTRIBUTING.md):",
            *(f"  - {problem}" for problem in problems),
            "Example: fix(readiness): keep a blocked gate blocked without evidence",
        ]
    )
    return 1


def _check_branch() -> int:
    branch = current_branch()
    if branch not in PROTECTED_BRANCHES:
        return 0
    _report(
        [
            f"Refusing to commit directly on '{branch}'.",
            "Create a branch and open a pull request (see CONTRIBUTING.md).",
        ]
    )
    return 1


def _check_range(revisions: str) -> int:
    failures = [
        (commit, subject, problems)
        for commit, subject in range_subjects(revisions)
        if (problems := subject_problems(subject))
    ]
    if not failures:
        return 0
    lines = [f"Commit subjects in {revisions} that break the convention:"]
    for commit, subject, problems in failures:
        lines.append(f"  {commit} {subject}")
        lines.extend(f"      - {problem}" for problem in problems)
    _report(lines)
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) == 2 and args[0] == "message":
        return _check_message(args[1])
    if args == ["branch"]:
        return _check_branch()
    if len(args) == 2 and args[0] == "range":
        return _check_range(args[1])
    _report(["usage: check_commit.py (message <file> | branch | range <revisions>)"])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
