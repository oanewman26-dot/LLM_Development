"""
config.py

Configuration contracts for AuroraLM.

Defines the PILOT / SMALL / FULL parameter ladders, the state-vector
contract, and the schema version saved with every checkpoint.

The 171-dimension names are NOT defined here. They are imported from
state_schema.py, which is the single source of truth and stays aligned
with EMOTION_WORDS in Aurora's live emotion_nodes.py.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from state_schema import (
    EMOTION_NAMES,
    EMOTION_COUNT,
    SCHEMA_VERSION,
    SCHEMA_FINGERPRINT,
)


# ---------------------------------------------------------------------------
# State schema version
# ---------------------------------------------------------------------------
# Re-exported from state_schema so existing imports keep working, but there
# is only one place it is actually defined.
STATE_SCHEMA_VERSION = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# AuroraState171 dimension contract
# ---------------------------------------------------------------------------
# Ordering comes straight from state_schema.EMOTION_NAMES. Never rebuild this
# list by hand — a reorder here silently teaches the model the wrong emotion.

STATE_DIMENSIONS: List[str] = list(EMOTION_NAMES)

if len(STATE_DIMENSIONS) != EMOTION_COUNT:
    raise RuntimeError(
        f"STATE_DIMENSIONS must have {EMOTION_COUNT} entries, "
        f"got {len(STATE_DIMENSIONS)}."
    )

# Thalamic activation scores are non-negative and are scaled by a single
# calibration ceiling (see state_schema.normalised_values), not per-dimension
# min/max. Every dimension therefore shares the same post-normalisation range.
STATE_RANGES: Dict[str, Tuple[float, float]] = {
    name: (0.0, 1.0) for name in STATE_DIMENSIONS
}

# A missing node means "did not fire", which is zero, not "unknown".
STATE_DEFAULTS: Dict[str, float] = {name: 0.0 for name in STATE_DIMENSIONS}

# Frozen from 59 genuine aurora-state-171/v1 captures using the maximum
# observed positive activation with no clipping. The aggregate, privacy-safe
# receipt is tracked at calibration/activation_calibration_v1.json. Any later
# score above this value must fail closed and trigger explicit recalibration.
ACTIVATION_CEILING: float = 0.78


# ---------------------------------------------------------------------------
# Regulator contract — Aurora's scalar drives, kept SEPARATE from the 171
# ---------------------------------------------------------------------------
# These are not emotion nodes. They come from the comparator, the pineal
# sleep-pressure accumulator, and the prefrontal confidence signal. They get
# their own small vector so they cannot be confused with thalamic activations.

REGULATOR_NAMES: Tuple[str, ...] = (
    "tension",
    "coherence",
    "sleep_pressure",
    "salience",
    "novelty",
    "confidence",
    "dominant_valence",
    "arousal",
    "dominance",
    "withdrawal_signal",
    "safety_margin",
)

REGULATOR_COUNT: int = len(REGULATOR_NAMES)

# Unlike the emotion nodes, these have meaningful per-dimension ranges and are
# clamped individually rather than scaled by one shared ceiling.
REGULATOR_RANGES: Dict[str, Tuple[float, float]] = {
    "tension":           (0.0, 1.0),
    "coherence":         (0.0, 1.0),
    "sleep_pressure":    (0.0, 1.0),
    "salience":          (0.0, 1.0),
    "novelty":           (0.0, 1.0),
    "confidence":        (0.0, 1.0),
    "dominant_valence": (-1.0, 1.0),
    "arousal":           (0.0, 1.0),
    "dominance":         (0.0, 1.0),
    "withdrawal_signal": (0.0, 1.0),
    "safety_margin":     (0.0, 1.0),
}

REGULATOR_DEFAULTS: Dict[str, float] = {
    name: 0.0 for name in REGULATOR_NAMES
}


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

@dataclass
class AuroraLMConfig:
    """
    Full configuration for AuroraLM.

    state_size=0 disables conditioning entirely and yields the control model.
    """

    # --- Identity ---
    name: str = "pilot"
    schema_version: str = STATE_SCHEMA_VERSION
    schema_fingerprint: str = SCHEMA_FINGERPRINT

    # --- Vocabulary ---
    vocab_size: int = 8192
    context_length: int = 512

    # --- Transformer core ---
    hidden_size: int = 256
    num_layers: int = 6
    num_attention_heads: int = 8
    num_kv_heads: int = 2          # grouped-query attention
    intermediate_size: int = 688   # SwiGLU hidden dim (~2.7x hidden_size)

    # --- State conditioning ---
    source_state_dim: int = EMOTION_COUNT   # always 171, from state_schema
    regulator_dim: int = REGULATOR_COUNT     # scalar drives, 0 = disabled
    state_size: int = 16                    # compressed latent dim; 0 = disabled
    state_hidden_dim: int = 64              # encoder trunk width
    state_dropout: float = 0.0

    # --- MoE (optional) ---
    num_experts: int = 0           # 0 = dense FFN, no MoE
    top_k_experts: int = 2
    expert_capacity_factor: float = 1.0

    # --- Normalisation ---
    rms_norm_eps: float = 1e-6

    # --- Initialisation ---
    init_std: float = 0.02

    # --- Training ---
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 100
    max_steps: int = 10_000
    batch_size: int = 32
    gradient_clip: float = 1.0

    # --- Checkpointing ---
    checkpoint_every: int = 1000

    def estimated_params(self) -> int:
        """Rough parameter count for logging."""
        # Embedding
        embed = self.vocab_size * self.hidden_size

        # Attention per layer (QKV + output)
        head_dim = self.hidden_size // self.num_attention_heads
        attn = (
            self.hidden_size * self.num_attention_heads * head_dim      # Q
            + self.hidden_size * self.num_kv_heads * head_dim * 2       # K + V
            + self.hidden_size * self.hidden_size                       # output
        )

        # FFN per layer (SwiGLU: gate + up + down)
        ffn = (
            self.hidden_size * self.intermediate_size * 2               # gate + up
            + self.intermediate_size * self.hidden_size                 # down
        )
        if self.num_experts > 0:
            # One independent SwiGLU per expert plus the token router.
            ffn_block = (
                self.num_experts * ffn
                + self.hidden_size * self.num_experts
                + self.num_experts
            )
        else:
            ffn_block = ffn

        # State conditioners per block (2 per block: attn + ffn)
        if self.state_size > 0:
            # StateConditioner MLP: state_size -> hidden_size -> 2*hidden_size
            cond = (
                self.state_size * self.hidden_size + self.hidden_size
                + self.hidden_size * (2 * self.hidden_size) + (2 * self.hidden_size)
            )
            cond_total = 2 * cond  # one for attn, one for ffn
        else:
            cond_total = 0

        # Norms per layer (2 RMSNorms per block)
        norms = 2 * self.hidden_size

        # Output head
        head = self.hidden_size * self.vocab_size
        final_norm = self.hidden_size

        # State encoder (if enabled)
        if self.state_size > 0:
            trunk_in = self.source_state_dim + self.regulator_dim
            state_enc = (
                trunk_in * self.state_hidden_dim + self.state_hidden_dim
                + self.state_hidden_dim * self.state_size + self.state_size
            )
            if self.num_experts > 0:
                state_enc += (
                    self.state_hidden_dim * self.num_experts + self.num_experts
                )
        else:
            state_enc = 0

        total = (
            embed
            + self.num_layers * (attn + ffn_block + norms + cond_total)
            + final_norm
            + head
            + state_enc
        )
        return total


# ---------------------------------------------------------------------------
# Predefined configs
# ---------------------------------------------------------------------------

PILOT = AuroraLMConfig(
    name="pilot",
    vocab_size=8192,
    context_length=512,
    hidden_size=256,
    num_layers=6,
    num_attention_heads=8,
    num_kv_heads=2,
    intermediate_size=688,
    state_size=16,
    state_hidden_dim=64,
    num_experts=0,
    batch_size=32,
    max_steps=10_000,
)

MOE_PILOT = AuroraLMConfig(
    **{
        **PILOT.__dict__,
        "name": "moe_pilot",
        "num_experts": 8,
        "top_k_experts": 2,
        "expert_capacity_factor": 1.0,
    }
)

SMALL = AuroraLMConfig(
    name="small",
    vocab_size=32000,
    context_length=2048,
    hidden_size=768,
    num_layers=12,
    num_attention_heads=12,
    num_kv_heads=4,
    intermediate_size=2048,
    state_size=16,
    state_hidden_dim=128,
    num_experts=0,
    batch_size=64,
    max_steps=50_000,
)

FULL = AuroraLMConfig(
    name="full",
    vocab_size=32000,
    context_length=4096,
    hidden_size=2048,
    num_layers=24,
    num_attention_heads=16,
    num_kv_heads=4,
    intermediate_size=5504,
    state_size=16,
    state_hidden_dim=256,
    num_experts=16,
    top_k_experts=2,
    batch_size=256,
    max_steps=100_000,
)

# Control model: identical to PILOT in every dimension except state is off.
# Built by copying PILOT's fields so the two can never drift apart.
CONTROL_PILOT = AuroraLMConfig(
    **{
        **PILOT.__dict__,
        "name": "control_pilot",
        "state_size": 0,
        "state_hidden_dim": 0,
        "num_experts": 0,
    }
)
