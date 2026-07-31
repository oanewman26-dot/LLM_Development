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
   python -m compileall -q .
   python -m pytest -q
   ```

5. Update the README or roadmap when a public contract or milestone changes.

Do not weaken readiness gates to make an experiment pass. A blocked gate is a
valid result when its evidence is absent.

## Licensing

Unless explicitly stated otherwise, intentionally submitted contributions are
provided under the repository's [Apache License 2.0](LICENSE), as described in
section 5 of that license. Contributors must have the right to submit their
work and must preserve relevant third-party notices.
