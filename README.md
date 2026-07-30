# AuroraLM

AuroraLM is an experimental decoder-only language model designed to receive
Aurora's live cognitive state as a real model input, rather than reducing that
state to descriptive prompt text.

The project asks a focused research question:

> Can a versioned 171-dimensional state signal improve model behaviour or
> expert-routing efficiency when compared with no state and shuffled-state
> controls?

Aurora's memory, appraisal, values, world model and action controls remain
outside the language model. AuroraLM is intended to be a replaceable voice
component within that wider system, not the whole mind.

> **Status:** research prototype. The public repository currently contains the
> runnable transformer core, state contract, tokenizer implementation and 18
> automated tests. It does not ship a trained model or claim production
> readiness.

## Current public milestone

- A 9,994,960-parameter dense PILOT transformer.
- RMSNorm, SwiGLU, rotary position embeddings and grouped-query attention.
- A canonical, fingerprinted `AuroraState171` schema.
- A `StateEncoder` that can combine 171 emotion-node activations with 11
  separate regulator values and compress them into a learned 16-dimensional
  latent. The current top-level model zero-fills the optional regulators.
- Adaptive layer-normalisation conditioning in every transformer block.
- A true stateless control configuration for later A/B experiments.
- An 8,192-token byte-level BPE implementation with fixed special-token IDs,
  deterministic save/load behaviour and artefact fingerprinting.
- Ten architecture smoke tests and eight tokenizer tests.

The published tests currently pass on CPU with PyTorch 2.13:

```text
Architecture smoke tests: 10 passed
Tokenizer tests:           8 passed
Total:                    18 passed
```

## Architecture

```mermaid
flowchart LR
    A["Aurora cognitive systems"] --> B["AuroraState171"]
    R["11 optional regulators<br/>(currently zero-filled)"] --> C["StateEncoder"]
    B --> C
    C --> D["16-dimensional state latent"]
    T["Token IDs"] --> E["Token embeddings"]
    E --> F["6 transformer blocks"]
    D --> F
    F --> G["RMSNorm"]
    G --> H["Vocabulary logits"]
```

The state encoder runs once per forward pass. Its compact latent is then
broadcast into the attention and feed-forward branches of every transformer
block through zero-initialised scale-and-shift conditioners.

This gives the project two important experimental properties:

1. At initialisation, random state and no state produce identical logits.
2. `CONTROL_PILOT` uses the same base architecture with state conditioning
   disabled, providing a clean comparison model.

The current public implementation is dense. Mixture-of-experts routing remains
a later experimental stage and should not be described as implemented.

## PILOT configuration

| Setting | Value |
| --- | ---: |
| Parameters | 9,994,960 |
| Vocabulary | 8,192 |
| Context length | 512 |
| Hidden size | 256 |
| Transformer blocks | 6 |
| Attention heads | 8 |
| Key/value heads | 2 |
| Feed-forward size | 688 |
| Source state dimensions | 171 |
| Regulator dimensions | 11 |
| Learned state latent | 16 |

`config.py` also contains `SMALL` and `FULL` planning configurations. They are
compute-ladder targets, not evidence that those models have been trained.

## Why structured state instead of prompt text?

A prompt can describe an internal state, but the model may ignore it, imitate
its language without using the signal, or confuse the description with user
content.

AuroraLM instead defines the state as a versioned numerical contract:

- the 171 emotion dimensions have a fixed order;
- missing nodes are represented explicitly as zero;
- invalid, negative and non-finite values are rejected;
- serialised records include a schema version and fingerprint;
- training and inference can reject incompatible state layouts;
- one fixed calibration ceiling preserves the difference between quiet and
  intense states.

The 11 regulator values remain separate from the 171 emotion nodes so signals
such as tension, coherence, sleep pressure and confidence cannot be mistaken
for emotions.

## Repository guide

| Path | Purpose |
| --- | --- |
| `NN.py` | Top-level `AuroraLM` model |
| `config.py` | PILOT, SMALL, FULL and stateless control contracts |
| `state_schema.py` | Canonical 171-dimensional state schema |
| `state_encoder.py` | State compression and optional routing-prior head |
| `transformer_block.py` | Pre-norm residual block with state conditioning |
| `rotary_position_embedding.py` | RoPE and grouped-query attention |
| `RSMnorm.py` | RMSNorm implementation |
| `feetforward.py` | SwiGLU feed-forward layer |
| `tokenizer.py` | Byte-level BPE training, loading and encoding CLI |
| `test_smoke.py` | Ten model and architecture checks |
| `test_tokenizer.py` | Eight tokenizer contract checks |
| `AuroraLM - State-Driven MoE Roadmap.txt` | Longer research roadmap |

## Quick start

### 1. Clone and create an environment

```bash
git clone https://github.com/oanewman26-dot/LLM_Development.git
cd LLM_Development
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Run the verification suite

```bash
python -m pytest test_smoke.py test_tokenizer.py -q
```

The two files can also be run directly:

```bash
python test_smoke.py
python test_tokenizer.py
```

### 3. Run a model forward pass

```python
import torch

from config import PILOT
from NN import AuroraLM

model = AuroraLM(PILOT)
tokens = torch.randint(0, PILOT.vocab_size, (2, 32))
aurora_state = torch.zeros(2, 171)

with torch.no_grad():
    logits = model(tokens, aurora_state)

print(logits.shape)  # torch.Size([2, 32, 8192])
```

This verifies the architecture only. Randomly initialised logits are not useful
language generation.

## Tokenizer CLI

Train an 8,192-token byte-level BPE tokenizer from clean UTF-8 text:

```bash
python tokenizer.py train \
  --input corpus.txt \
  --output artifacts/aurora_bpe_8192_v1 \
  --vocab-size 8192 \
  --min-frequency 2
```

Inspect a saved artefact:

```bash
python tokenizer.py inspect artifacts/aurora_bpe_8192_v1
```

Encode a short string:

```bash
python tokenizer.py encode artifacts/aurora_bpe_8192_v1 "Aurora notices the change."
```

Training corpora, private journals, checkpoints and generated artefacts should
be reviewed carefully and kept out of commits unless they are intentionally
licensed and suitable for publication.

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

## Roadmap

The longer technical plan is available in
[AuroraLM - State-Driven MoE Roadmap](./AuroraLM%20-%20State-Driven%20MoE%20Roadmap.txt).

Near-term public milestones:

- publish deterministic dataset preparation after privacy review;
- publish dense training and inference contracts;
- freeze a representative tokenizer artefact;
- run controlled base-model training;
- gather genuine state-aligned turns;
- calibrate the state activation ceiling;
- run no-state, correct-state and shuffled-state ablations;
- attempt MoE routing only if the dense PILOT produces useful evidence.

## Author

Built by [Omari Newman](https://www.linkedin.com/in/omari-newman/).

- GitHub: [@oanewman26-dot](https://github.com/oanewman26-dot)
- LinkedIn: [linkedin.com/in/omari-newman](https://www.linkedin.com/in/omari-newman/)
