import torch
import torch.nn as nn

from config import AuroraLMConfig
from RSMnorm import RMSNorm
from transformer_block import TransformerBlock
from state_encoder import StateEncoder


class AuroraLM(nn.Module):
    """
    AuroraLM: a decoder-only transformer with Aurora's internal state
    wired into every block via adaptive layer norm.

    Architecture:
        tokens -> embedding -> [TransformerBlock] -> RMSNorm -> logits
                              ^
                              |
                    aurora_state (171-dim)
                              |
                         StateEncoder
                              |
                    state_latent (broadcast to all blocks)

    When aurora_state is None or config.state_size == 0, the model is
    exactly a vanilla transformer — the zero-init baseline property holds.
    """

    def __init__(self, config: AuroraLMConfig):
        super().__init__()

        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size
        )

        # State encoder: compresses 171-dim source state to block latents
        if config.state_size > 0:
            self.state_encoder = StateEncoder(
                source_dim=171,
                state_size=config.state_size,
                hidden_dim=config.state_hidden_dim,
                num_experts=config.num_experts,
                dropout=config.state_dropout,
            )
        else:
            self.state_encoder = None

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.output = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )

    def forward(self, tokens, aurora_state=None):
        """
        Args:
            tokens: (batch, seq) — token indices
            aurora_state: (batch, 171) or None — raw Aurora emotional state

        Returns:
            logits: (batch, seq, vocab_size)
        """
        x = self.embedding(tokens)

        # Encode state once, broadcast latent to every block
        state_latent = None
        if self.state_encoder is not None and aurora_state is not None:
            state_latent, _ = self.state_encoder(aurora_state)

        for block in self.blocks:
            x = block(x, state_latent)

        x = self.norm(x)
        logits = self.output(x)

        return logits

    def count_parameters(self):
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
