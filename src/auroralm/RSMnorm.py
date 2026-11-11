import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalisation.

    Used in modern LLM architectures.
    Normalises the magnitude of hidden states
    while preserving their direction.

    Formula:

    x = x / RMS(x) * weight

    where:

    RMS(x) = sqrt(mean(x²) + eps)
    """

    def __init__(
        self,
        hidden_size,
        eps=1e-6
    ):

        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(hidden_size)
        )


    def forward(self, x):

        # Keep computation in higher precision
        # for numerical stability

        input_dtype = x.dtype

        x_float = x.float()


        rms = torch.sqrt(
            x_float.pow(2).mean(
                dim=-1,
                keepdim=True
            )
            +
            self.eps
        )


        x_norm = (
            x_float / rms
        )


        return (
            x_norm
            .to(input_dtype)
            *
            self.weight
        )