import torch
import torch.nn as nn
import torch.nn.functional as F

from rotary_position_embedding import GroupedQueryAttention
from feetforward import SwiGLU
from RSMnorm import RMSNorm


class StateConditioner(nn.Module):
    """
    Maps Aurora's state vector to per-block scale and shift.

    Adaptive layer norm (adaLN), repurposed from diffusion transformers.
    Zero-initialised so the model is exactly a vanilla transformer at step 0.
    """

    def __init__(self, state_size, hidden_size):
        super().__init__()

        self.state_size = state_size
        self.hidden_size = hidden_size

        # Small MLP: state vector -> scale + shift
        # Output is 2 * hidden_size: first half is scale, second half is shift
        self.mlp = nn.Sequential(
            nn.Linear(state_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

        # Zero-initialise the final projection so scale=1, shift=0 at start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, aurora_state):
        """
        Args:
            x: (batch, seq, hidden_size) — normalised hidden states
            aurora_state: (batch, state_size) or None

        Returns:
            conditioned x with same shape
        """
        if aurora_state is None or self.state_size == 0:
            return x

        # (batch, 2 * hidden_size)
        params = self.mlp(aurora_state)

        # Split into scale and shift
        scale, shift = params.chunk(2, dim=-1)

        # Expand to (batch, 1, hidden_size) for broadcasting over seq
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)

        return x * (1 + scale) + shift


class TransformerBlock(nn.Module):
    """
    Pre-norm residual transformer block with state conditioning.

    Architecture:
        x = x + attn(cond(norm(x)))
        x = x + ffn(cond(norm(x)))

    The conditioner applies adaptive layer norm after RMSNorm and before
    the attention / feed-forward sublayer. Because it multiplies every
    hidden state, the network cannot ignore Aurora's state.
    """

    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.state_size = getattr(config, "state_size", 16)

        # Pre-norm layers
        self.attn_norm = RMSNorm(config.hidden_size)
        self.ffn_norm = RMSNorm(config.hidden_size)

        # Attention + FFN
        self.attn = GroupedQueryAttention(config)
        self.ffn = SwiGLU(config)

        # State conditioners — one for each sublayer
        if self.state_size > 0:
            self.attn_cond = StateConditioner(self.state_size, config.hidden_size)
            self.ffn_cond = StateConditioner(self.state_size, config.hidden_size)
        else:
            self.attn_cond = None
            self.ffn_cond = None

    def forward(self, x, aurora_state=None):
        """
        Args:
            x: (batch, seq, hidden_size)
            aurora_state: (batch, state_size) or None

        Returns:
            x: (batch, seq, hidden_size)
        """
        # --- Attention branch ---
        normed = self.attn_norm(x)

        if self.attn_cond is not None:
            normed = self.attn_cond(normed, aurora_state)

        x = x + self.attn(normed)

        # --- Feed-forward branch ---
        normed = self.ffn_norm(x)

        if self.ffn_cond is not None:
            normed = self.ffn_cond(normed, aurora_state)

        x = x + self.ffn(normed)

        return x
