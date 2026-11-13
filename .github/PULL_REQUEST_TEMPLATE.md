<!-- Title: a Conventional Commits subject, e.g. "fix(scope): what changed".
     The only pull request is `staging` into `main`; no other branch is used. -->

## What and why

<!-- What this changes and the reason for it. One subsystem per pull request. -->

## Behaviour

<!-- Say whether this changes a public contract, a readiness gate or a
     reproducibility pin (dataset revision, seed), or write "none". -->

- [ ] No readiness gate is weakened to make an experiment pass

## Checks

- [ ] `python -m pytest -q` passes locally
- [ ] Tests added or updated for behavioural changes
- [ ] Every commit subject follows Conventional Commits
- [ ] No private data, runtime state, weights or credentials are included
- [ ] Docs updated where a contract or procedure changed
