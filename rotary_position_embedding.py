import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    """
    Splits the last dimension into two halves
    and rotates them.
    """

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """
    Rotary positional embeddings (RoPE)
    """

    def __init__(self, dim, max_position_embeddings=4096, theta=10000):

        super().__init__()

        self.dim = dim

        # Build frequencies for dim/2, then duplicate to match full dim
        # This matches the rotate_half convention: each pair of dimensions
        # shares the same frequency
        half_dim = dim // 2
        frequencies = 1.0 / (
            theta ** (
                torch.arange(0, half_dim).float()
                / half_dim
            )
        )

        # Duplicate frequencies: [f0, f1, ...] -> [f0, f0, f1, f1, ...]
        frequencies = frequencies.repeat_interleave(2)

        positions = torch.arange(
            max_position_embeddings
        ).float()

        angles = torch.outer(
            positions,
            frequencies
        )

        self.register_buffer(
            "cos",
            angles.cos(),
            persistent=False
        )

        self.register_buffer(
            "sin",
            angles.sin(),
            persistent=False
        )


    def forward(self, x, seq_length):

        cos = self.cos[:seq_length]
        sin = self.sin[:seq_length]

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        return (
            x * cos
            +
            rotate_half(x) * sin
        )



class GroupedQueryAttention(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.hidden_size = config.hidden_size

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads

        self.head_dim = (
            config.hidden_size
            //
            config.num_attention_heads
        )

        assert (
            self.num_heads % self.num_kv_heads == 0
        ), "Heads must divide evenly"


        self.query = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False
        )

        self.key = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False
        )

        self.value = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False
        )


        self.output = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=False
        )


        self.rope = RotaryEmbedding(
            self.head_dim,
            config.context_length
        )


    def repeat_kv(self, x):

        """
        Expands KV heads to match query heads.

        Example:

        KV heads = 4
        Query heads = 16

        Each KV head gets reused 4 times.
        """

        repeat_factor = (
            self.num_heads
            //
            self.num_kv_heads
        )

        return x.repeat_interleave(
            repeat_factor,
            dim=1
        )


    def forward(self, x):

        batch, seq, _ = x.shape


        q = self.query(x)
        k = self.key(x)
        v = self.value(x)


        q = q.view(
            batch,
            seq,
            self.num_heads,
            self.head_dim
        )

        k = k.view(
            batch,
            seq,
            self.num_kv_heads,
            self.head_dim
        )

        v = v.view(
            batch,
            seq,
            self.num_kv_heads,
            self.head_dim
        )


        # Move heads before sequence

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)


        # Apply rotary embeddings

        q = self.rope(q, seq)
        k = self.rope(k, seq)


        # Expand KV heads

        k = self.repeat_kv(k)
        v = self.repeat_kv(v)


        # Attention scores

        scores = (
            q @ k.transpose(-2,-1)
        ) / (
            self.head_dim ** 0.5
        )


        # Causal mask

        mask = torch.triu(
            torch.ones(
                seq,
                seq,
                device=x.device
            ),
            diagonal=1
        ).bool()


        scores = scores.masked_fill(
            mask,
            float("-inf")
        )


        weights = F.softmax(
            scores,
            dim=-1
        )


        output = weights @ v


        output = output.transpose(1,2)

        output = output.contiguous().view(
            batch,
            seq,
            self.hidden_size
        )


        return self.output(output)
