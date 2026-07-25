import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network.

    Used in modern LLM architectures.

    Formula:

    output = down(
        silu(gate(x)) * up(x)
    )
    """

    def __init__(self, config):

        super().__init__()

        self.gate = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False
        )

        self.up = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False
        )

        self.down = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False
        )


    def forward(self, x):

        gate = self.gate(x)
        up = self.up(x)

        activated = (
            F.silu(gate)
            *
            up
        )

        output = self.down(
            activated
        )

        return output