# AuroraLM status

These sections were part of the root `README.md` until 2026-11-13 and were
moved here unchanged. They are dated records: the test counts below cover the
65 model, data, training and MoE contract tests, not the repository-policy
tests added since.

## Current public milestone

- A 9,994,960-parameter dense PILOT transformer.
- RMSNorm, SwiGLU, rotary position embeddings and grouped-query attention.
- A canonical, fingerprinted `AuroraState171` schema.
- A `StateEncoder` that can combine 171 emotion-node activations with 11
  separate regulator values and compress them into a learned 16-dimensional
  latent. The current top-level model zero-fills the optional regulators.
- Adaptive layer-normalisation conditioning in every transformer block.
- A true stateless control configuration for later A/B experiments.
- An optional 8-expert, top-2 `MOE_PILOT` path with token routing,
  causal capacity handling, overflow telemetry and function-preserving dense
  expert cloning. It is implemented and tested, but not trained.
- An 8,192-token byte-level BPE implementation with fixed special-token IDs,
  deterministic save/load behaviour and artefact fingerprinting.
- Sixty-five tests across architecture, data, training, inference, readiness,
  activation calibration and MoE contracts.

The published tests currently pass on CPU with PyTorch 2.13:

```text
Architecture smoke tests: 10 passed
Tokenizer tests:           9 passed
Dataset tests:             5 passed
Sampler tests:             3 passed
Training tests:            8 passed
Inference tests:           7 passed
Readiness-gate tests:      3 passed
Activation calibration:   3 passed
MoE router tests:          6 passed
Expert dispatch tests:     5 passed
MoE integration tests:     6 passed
Total:                    65 passed
```

## Verified development status — 2026-08-01

- The public AuroraLM suite passes all 65 tests.
- Aurora's private suite passes all 109 tests on local checkpoint `741ce9c`.
  That checkpoint contains the exact 11-regulator logger change and has not
  yet been merged into Aurora's `dev` branch or pushed.
- Aurora and AuroraLM have an exact cross-repository match for all 171 emotion
  dimensions, the state-schema fingerprint, and all 11 regulator names and
  ranges.
- The phase-two checkpoint contains 9,994,960 finite parameters and verifies
  at SHA-256
  `8056f7455845bae761625f81a42245f71b5ddd536584c3f583ca91bd69ffa787`.
- Eighteen phase-two qualitative generations produce 412 distinct candidate
  16-word shingles; none match the canonical training documents exactly.
- A private recovery audit restored 96 historical files byte-for-byte into a
  git-ignored local packet without changing either live worktree. Recovered
  journals, memories, vault data and runtime logs remain private and outside
  this repository.
- Gate 1 remains `PENDING` because the blinded semantic scorecard is
  incomplete. Gate 2 remains `BLOCKED` because the canonical dataset still has
  zero post-fix exact state-aligned turns. State-conditioned training, the
  correct/no-state/shuffled-state ablation, and state-influenced MoE routing
  remain unauthorised.
