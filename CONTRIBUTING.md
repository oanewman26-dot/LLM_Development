# Contributing to AuroraLM

AuroraLM is a gated research prototype. Contributions should preserve its
reproducibility, privacy boundaries, and distinction between implemented
mechanics and demonstrated research results.

## Before opening a pull request

1. Keep private Aurora records, journals, training corpora, checkpoints,
   credentials, and machine-specific paths out of the repository.
2. Keep public dataset revisions and random seeds pinned where reproducibility
   depends on them.
3. Add or update tests for behavioural changes.
4. Run:

   ```bash
   python -m pip check
   python -m ruff format .
   python -m ruff check .
   python -m mypy
   python -m compileall -q .
   python -m pytest -q
   ```

   `python -m pre_commit install` runs the format, lint, type and compile
   checks on every commit, checks the commit message, and runs the test
   suite before a push. `print()` is
   allowed only in the modules with a command-line entry point (listed in
   `pyproject.toml`).

5. Update the README or roadmap when a public contract or milestone changes.

## Commits and branches

- **Branch from `staging`**, named `<type>/<short-topic>` (for example
  `fix/readiness-gate-evidence`). Open a pull request back into `staging`;
  `staging` is the integration branch and reaches `main` only through a
  pull request. Nothing is committed directly to `main`: the pre-commit hook
  refuses it locally, and `main` should stay protected on GitHub.
- **Write [Conventional Commits](https://www.conventionalcommits.org/)**:
  `type(optional-scope): summary`, with the type one of `feat`, `fix`,
  `docs`, `test`, `refactor`, `style`, `perf`, `build`, `ci`, `chore` or
  `revert`. Keep the subject to 72 characters with no closing full stop, and
  put the reasoning in the body after a blank line. Add `!` before the colon
  for a breaking change. The `commit-msg` hook and the "Commit messages" CI
  job both run `scripts/check_commit.py`; the pull request title follows the
  same form.
- **One subsystem per commit and per pull request.** Split a change that
  spans several modules into a commit for each, and keep
  mechanical changes (formatting, renames, moves) apart from behavioural
  ones. If a commit needs "and" in its subject, it is probably two commits.
- Do not leave work in long-lived stashes or `recovery/*` branches. Push a
  draft pull request instead, so unfinished work is visible and reviewed.

Do not weaken readiness gates to make an experiment pass. A blocked gate is a
valid result when its evidence is absent.

## Licensing

Unless explicitly stated otherwise, intentionally submitted contributions are
provided under the repository's [Apache License 2.0](LICENSE), as described in
section 5 of that license. Contributors must have the right to submit their
work and must preserve relevant third-party notices.
