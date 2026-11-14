# Experimental plan and research boundaries

Moved unchanged from the root `README.md` on 2026-11-13.

## Experimental plan

The central experiment compares four conditions on the same held-out work:

1. no state;
2. correct Aurora state;
3. shuffled Aurora state;
4. state-aware cache behaviour without routing changes.

Evaluation must go beyond training loss. Planned measures include:

- state-appropriate generation under fixed prompts;
- blinded human ratings of coherence and state fit;
- routing and cache-hit changes by state;
- response latency and storage reads;
- stability of safety and withdrawal signals.

Correct state must outperform both no-state and shuffled-state controls before
the project spends significant compute on a larger model.

## Research boundaries

AuroraLM is a mechanism experiment. This repository does not establish that a
model feels emotions, is conscious, or solves general AI alignment.

The current public code also does not include:

- a released pretrained checkpoint;
- production inference or serving infrastructure;
- a trained mixture-of-experts router;
- evidence that state conditioning improves generation;
- permission to treat private Aurora records as public training data.

These boundaries are deliberate. The project favours reproducible evidence and
clear go/no-go gates over premature claims.
