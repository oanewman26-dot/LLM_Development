"""Commit message and branch policy (shared with the Aurora repository)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "check_commit", ROOT / "scripts" / "check_commit.py"
)
assert _spec is not None and _spec.loader is not None
check_commit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_commit)


@pytest.mark.parametrize(
    "subject",
    [
        "feat: add relational tension from appraisal events",
        "fix(readiness): keep a blocked gate blocked without evidence",
        "refactor(core/agent)!: drop the legacy state reader",
        "build(deps): bump numpy from 2.5.1 to 2.5.4",
        "docs: record Phase 5 status in the repo alignment plan",
        "Merge pull request #58 from owner/staging",
        'Revert "heartbeat: idle measures the comparator"',
        "fixup! fix(readiness): keep a blocked gate blocked without evidence",
    ],
)
def test_conforming_subjects_are_accepted(subject):
    assert check_commit.subject_problems(subject) == []


@pytest.mark.parametrize(
    "subject",
    [
        "Update roadmap",
        "tension: add relational tension from appraisal events",
        "Fix: capitalised type",
        "fix:missing space",
        "fix : space before the colon",
        "fix(): empty scope",
        "fix: ends with a full stop.",
        "feat: " + "x" * 80,
        "",
        "   ",
    ],
)
def test_nonconforming_subjects_are_rejected(subject):
    assert check_commit.subject_problems(subject)


def test_subject_at_the_length_limit_is_accepted():
    subject = "feat: " + "x" * (check_commit.MAX_SUBJECT_LENGTH - len("feat: "))
    assert len(subject) == check_commit.MAX_SUBJECT_LENGTH
    assert check_commit.subject_problems(subject) == []
    assert check_commit.subject_problems(subject + "x")


def test_message_ignores_git_comments_and_the_diff_below_the_scissors():
    text = (
        "# Please enter the commit message for your changes.\n"
        "\n"
        "fix: close the session before world entry\n"
        "\n"
        "The body explains why.\n"
        "# ------------------------ >8 ------------------------\n"
        "diff --git a/x b/x\n"
        "+Not A Subject\n"
    )
    assert check_commit.message_problems(text) == []


def test_message_requires_a_blank_line_before_the_body():
    problems = check_commit.message_problems("fix: one thing\nand a body\n")
    assert len(problems) == 1 and "second line" in problems[0]


def test_empty_message_is_rejected():
    assert check_commit.message_problems("# only a comment\n\n")


def test_message_file_check_returns_an_exit_status(tmp_path, capsys):
    good = tmp_path / "good"
    good.write_text("test: cover the commit policy\n", encoding="utf-8")
    bad = tmp_path / "bad"
    bad.write_text("WIP stuff\n", encoding="utf-8")
    assert check_commit.main(["message", str(good)]) == 0
    assert check_commit.main(["message", str(bad)]) == 1
    assert "Commit message rejected" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("branch", "status"),
    [("main", 1), ("staging", 0), ("feat/shelter-action", 0), (None, 0)],
)
def test_direct_commits_are_refused_only_on_protected_branches(
    monkeypatch, branch, status
):
    monkeypatch.setattr(check_commit, "current_branch", lambda: branch)
    assert check_commit.main(["branch"]) == status


def test_range_check_reports_each_offending_commit(monkeypatch, capsys):
    monkeypatch.setattr(
        check_commit,
        "range_subjects",
        lambda revisions: [("abc1234", "fix: a good one"), ("def5678", "bad one")],
    )
    assert check_commit.main(["range", "origin/main..HEAD"]) == 1
    err = capsys.readouterr().err
    assert "def5678 bad one" in err and "abc1234" not in err


def test_range_check_passes_when_every_subject_conforms(monkeypatch):
    monkeypatch.setattr(
        check_commit, "range_subjects", lambda revisions: [("abc1234", "ci: add job")]
    )
    assert check_commit.main(["range", "origin/main..HEAD"]) == 0


def test_unknown_arguments_print_usage(capsys):
    assert check_commit.main([]) == 2
    assert "usage:" in capsys.readouterr().err
