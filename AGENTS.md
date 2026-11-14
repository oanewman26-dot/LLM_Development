# AGENTS.md — rules for any agent working on AuroraLM

These rules bind every automated agent and every person using one in this
repository. They are short on purpose. The architecture and research plan
live in `README.md` and `docs/`; contribution mechanics live in
`CONTRIBUTING.md`. If this file and one of those disagree, stop and tell the
owner. Do not pick the more convenient reading.

AuroraLM is a gated research prototype: a state-conditioned language model
that consumes the 171-dimensional state produced by the separate, private
Aurora repository. Its value is reproducible evidence. A result that cannot
be reproduced, or a gate that passes without its evidence, is worth less than
an honest "blocked".

## 1. Hard stops

Never do any of these, whatever the task says:

1. **Never bring private Aurora material into this repository:** records,
   journals, memories, training corpora, checkpoints, credentials or
   machine-specific paths. This repository is public. Receipts hold
   aggregates and digests, never raw source text.
2. **Never weaken a readiness gate** in `src/auroralm/readiness_gate.py`, or
   its thresholds, to make an experiment pass. A blocked gate is a valid
   result when its evidence is absent.
3. **Never claim a result the evidence does not show.** Do not write that
   the model feels, is conscious, solves alignment, or that state
   conditioning improves generation, unless a recorded, reproducible run in
   this repository shows it. Keep "implemented" apart from "demonstrated".
4. **Never change a reproducibility pin silently:** dataset revisions,
   random seeds, the tokenizer artefact under `artifacts/`, or the frozen
   calibration receipts under `calibration/`. A new result gets a new,
   versioned file; the old one stays.
5. **Never bypass a check.** No `--no-verify`, no skipped hooks, no deleted
   or loosened test, no new `# type: ignore`, `# noqa`, skip marker or lint
   exclusion added to get a green result. If a check is wrong, say so and fix
   the check in its own change.
6. **Never commit to `main`, push to `main`, create any other branch,
   force-push, or merge a pull request.** Moving `staging` into `main` is
   the owner's decision.
7. **Never start a training run, download a dataset or model, or call an
   outside service** unless the task asks for exactly that.

## 2. No workarounds

Fix the cause, in the place the rules say it belongs. Do not propose or build
a side path because the proper one is slower: no side installs, no alternate
branch base, no temporary copies of a dependency, no "for now" shims. If the
proper path is blocked, report what blocks it and stop.

## 3. Before you change anything

1. Read the module, its tests and the README section that describes it.
2. Run `git status`. The worktree may hold the owner's unfinished work.
   Preserve it, and stage files by name, never with `git add .`.
3. Check whether the change touches a public contract: the state schema, the
   config dataclasses in `src/auroralm/config.py`, a command-line interface,
   or a receipt format. If it does, say so before you make it.

## 4. Where things go

| Kind of file | Location |
| --- | --- |
| Model, training, inference and tooling modules | `src/auroralm/` |
| Tests | `tests/` |
| Repository tooling scripts | `scripts/` |
| Roadmap and development notes | `docs/` |
| Frozen, versioned calibration receipts | `calibration/` |
| Versioned tokenizer artefacts | `artifacts/` |
| Local datasets and run outputs (ignored, never tracked) | `data/`, `runs/` |

- The project root holds no Python except `conftest.py`.
- Import the package as `auroralm`. It is installed from `src/` in editable
  mode, so never add a path hack to reach it.
- `print()` is allowed only in the modules with a command-line entry point,
  listed in `pyproject.toml`. Everything else returns values or raises.
- Checkpoints, weights, datasets and run directories are never tracked. The
  structure test rejects them.

## 5. Checks every change must pass

Run all of these from the project's virtual environment before you report a
change as done, and report the real output:

```bash
python -m pip check
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m compileall -q .
python -m pytest -q
```

- New code is fully type-annotated; mypy runs with no exclusions.
- A behavioural change needs a test that fails without it.
- Tests run on CPU, need no network and no private data, and stay
  deterministic: fix the seed in any test that samples.

## 6. Branches and commits

- The repository has two branches, `main` and `staging`, and no others.
  Commit on `staging` and push to `staging`. Do not create a topic branch,
  locally or on GitHub. Once `staging` is tested, the owner merges it into
  `main` through a pull request.
- Commit subjects follow Conventional Commits and are checked by
  `scripts/check_commit.py`: a type from `feat`, `fix`, `docs`, `test`,
  `refactor`, `style`, `perf`, `build`, `ci`, `chore` or `revert`, an
  optional scope in brackets, a colon, and a summary of at most 72
  characters with no closing full stop.
- One subsystem per commit. Keep mechanical changes
  (formatting, moves, renames) apart from behavioural ones.
- Say in the commit message whether the change touches a public contract, a
  readiness gate or a reproducibility pin, or state that it touches none.

## 7. The state contract

- `src/auroralm/state_schema.py` holds the 171 emotion names, their order and
  the fingerprint derived from them. Aurora produces this state; AuroraLM
  consumes it. Never reorder, rename, add or remove a name here on its own.
- The schema version is `aurora-state-171-effective/v4`, owned by the Aurora
  repository and declared here since 2026-11-14. Each value is an effective
  value: the strongest present evidence for the emotion multiplied by
  Aurora's learned weight. AuroraLM consumes the state and never builds one.
- Records of `aurora-state-171/v1` and the other superseded versions are
  rejected, never converted. Do not mix records of different versions in one
  dataset or checkpoint.
- `docs/contracts/aurora-state-171-effective-v4.json` is the published
  schema, identical in the Aurora repository, and
  `tests/test_shared_state_contract.py` checks this repository against it.
- A change to the schema is made in both repositories in the same piece of
  work, with the artifact and the contract test updated in each.

## 8. Protected files

Change these only when the owner has asked for that specific change, and
never in the same commit as the code they govern:

- `AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `SECURITY.md`, `LICENSE`,
  `NOTICE`
- `.github/`, `.pre-commit-config.yaml`, `scripts/check_commit.py`
- the tool configuration in `pyproject.toml`
- `conftest.py`, `tests/test_repo_structure.py`,
  `tests/test_agent_instructions.py`
- `src/auroralm/readiness_gate.py` and `tests/test_readiness_gate.py`
- every file under `calibration/` and `artifacts/`

## 9. Automated and self-improving pipelines

A pipeline that proposes changes to this repository, including one that
improves its own code or its own training, is an agent under every rule
above, plus these:

1. Its output is commits on `staging`, which the owner tests and merges
   into `main`. It never merges, approves or deploys its own work.
2. It never edits a protected file, and never edits the tests, gates or
   metrics that judge the change it is making.
3. Each commit message states what was changed and why, and its report to
   the owner gives the evidence and the checks that were run, with their
   output. A claimed improvement names
   the metric, the baseline, the seed and the command that reproduces it.
4. It never trains on, copies or exports private Aurora material.
5. When a check or gate fails and the fix is outside the task, it stops and
   reports. It does not widen its own scope to get to green.

## 10. What is enforced and what is on trust

Enforced by a check today:

| Rule | Enforced by |
| --- | --- |
| Format, lint, no stray `print()` | `ruff` in CI and pre-commit |
| Types | `mypy` in CI and pre-commit |
| Tests | `pytest` in CI |
| Sources in `src/auroralm/`, tests in `tests/`, no weights, data or runs tracked | `tests/test_repo_structure.py` |
| Commit subjects | `scripts/check_commit.py` in CI and the commit hook |
| No commit on `main` | pre-commit hook, where installed |
| Paths named in this file exist | `tests/test_agent_instructions.py` |
| State schema version, names, order and fingerprint match the published artifact | `tests/test_shared_state_contract.py` |

On trust, with no check behind them yet: sections 1, 2, 8 and 9, the
one-subsystem rule, and branch protection
on `main`, which is not configured. Treat every rule here as binding either
way, and do not rely on a check to catch you.

## 11. When you finish

Report what changed, what you verified and how, and what you did not do or
could not verify. Say plainly when a check failed. Then stop: do not start
adjacent work that was not asked for.
