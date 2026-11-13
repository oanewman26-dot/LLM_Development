"""
state_encoder.py

Bridge between Aurora's live state and AuroraLM's transformer blocks.

Two inputs, deliberately kept separate:

    1. aurora_state  (batch, 171) — thalamic emotion node activations,
       normalised by ONE shared calibration ceiling.
    2. regulators    (batch, 11)  — scalar drives from the comparator,
       pineal accumulator, and prefrontal confidence signal. These have
       their own per-dimension ranges and are NOT emotion nodes.

They are concatenated before the trunk rather than summed or interleaved,
so a regulator can never be mistaken for an emotion node and the schema
fingerprint stays meaningful.

Outputs:
    - state_latent:  (batch, state_size)  -> adaptive layer norm per block
    - routing_prior: (batch, num_experts) -> expert warming bias (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateEncoder(nn.Module):
    """Compresses Aurora's state to task-specific latents."""

    def __init__(
        self,
        source_dim=171,
        regulator_dim=11,
        state_size=16,
        hidden_dim=64,
        num_experts=0,
        dropout=0.0,
    ):
        super().__init__()

        self.source_dim = source_dim
        self.regulator_dim = regulator_dim
        self.state_size = state_size
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts

        # Trunk input is the two vectors concatenated
        self.trunk_in = source_dim + regulator_dim

        # Simple MLP trunk. SiLU rather than SwiGLU gating to guarantee
        # gradient flow through a small encoder.
        self.trunk = nn.Sequential(
            nn.Linear(self.trunk_in, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

        # --- Heads ---
        self.state_head = nn.Linear(hidden_dim, state_size, bias=True)

        if num_experts > 0:
            self.routing_head = nn.Linear(hidden_dim, num_experts, bias=True)
            # Soft init: routing prior starts near uniform
            nn.init.normal_(self.routing_head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.routing_head.bias)
        else:
            self.routing_head = None

        # Zero-init the state head so the baseline property holds: at step 0
        # a real state and no state produce identical logits.
        nn.init.zeros_(self.state_head.weight)
        nn.init.zeros_(self.state_head.bias)

    # -----------------------------------------------------------------------

    def _assemble(self, aurora_state, regulators):
        """Concatenate the two inputs, filling regulators with zeros if absent.

        NOTE: a zero regulator vector is indistinguishable from "regulators
        not supplied". If that distinction matters later, add an explicit
        presence flag rather than relying on the zeros.
        """
        if self.regulator_dim == 0:
            return aurora_state

        if regulators is None:
            regulators = aurora_state.new_zeros(
                aurora_state.shape[0], self.regulator_dim
            )

        if regulators.shape[-1] != self.regulator_dim:
            raise ValueError(
                f"Expected {self.regulator_dim} regulators, "
                f"got {regulators.shape[-1]}."
            )

        return torch.cat([aurora_state, regulators], dim=-1)

    def forward(self, aurora_state, regulators=None):
        """
        Args:
            aurora_state: (batch, 171) or None
            regulators:   (batch, 11), or None to use zeros

        Returns:
            state_latent:  (batch, state_size)  or None
            routing_prior: (batch, num_experts) or None
        """
        if aurora_state is None:
            return None, None

        x = self._assemble(aurora_state, regulators)
        hidden = self.trunk(x)

        state_latent = self.state_head(hidden)

        if self.routing_head is not None:
            routing_prior = self.routing_head(hidden)
        else:
            routing_prior = None

        return state_latent, routing_prior

    # -----------------------------------------------------------------------

    def get_routing_weights(self, aurora_state, regulators=None, temperature=1.0):
        """Soft expert weights for cache warming. Sums to 1 per batch item."""
        _, routing_prior = self.forward(aurora_state, regulators)
        if routing_prior is None:
            return None
        return F.softmax(routing_prior / temperature, dim=-1)
