"""Token-level routing for AuroraLM's sparse mixture-of-experts blocks.

The token representation is always the primary routing signal.  Aurora's
batch-level state prior is optional and can only contribute a bounded additive
bias to the token router logits.  A scale of zero keeps the state pathway in
the computation graph without allowing it to change an expert decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RouterOutput:
    """Routing decisions and differentiable regularisation terms."""

    token_logits: torch.Tensor
    state_bias: torch.Tensor
    routing_logits: torch.Tensor
    probabilities: torch.Tensor
    expert_indices: torch.Tensor
    expert_weights: torch.Tensor
    expert_counts: torch.Tensor
    load_balancing_loss: torch.Tensor
    router_z_loss: torch.Tensor
    entropy: torch.Tensor


class MoERouter(nn.Module):
    """Route every token to a normalized top-k set of experts."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        *,
        top_k: int = 2,
        state_prior_scale: float = 0.0,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()

        if hidden_size < 1:
            raise ValueError("hidden_size must be at least 1.")
        if num_experts < 1:
            raise ValueError("num_experts must be at least 1.")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts.")
        if state_prior_scale < 0.0:
            raise ValueError("state_prior_scale cannot be negative.")
        if init_std <= 0.0:
            raise ValueError("init_std must be positive.")

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.state_prior_scale = float(state_prior_scale)

        self.token_router = nn.Linear(hidden_size, num_experts, bias=True)
        nn.init.normal_(self.token_router.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.token_router.bias)

    def _bounded_state_bias(
        self,
        hidden_states: torch.Tensor,
        state_prior: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, sequence, _ = hidden_states.shape
        if state_prior is None:
            return hidden_states.new_zeros(
                (batch, sequence, self.num_experts),
                dtype=torch.float32,
            )

        if state_prior.ndim == 2:
            if state_prior.shape != (batch, self.num_experts):
                raise ValueError(
                    "Expected state_prior [batch, num_experts], got "
                    f"{tuple(state_prior.shape)}."
                )
            state_prior = state_prior.unsqueeze(1)
        elif state_prior.ndim == 3:
            if state_prior.shape != (batch, sequence, self.num_experts):
                raise ValueError(
                    "Expected state_prior [batch, sequence, num_experts], got "
                    f"{tuple(state_prior.shape)}."
                )
        else:
            raise ValueError(
                "state_prior must have shape [batch, num_experts] or "
                "[batch, sequence, num_experts]."
            )

        state_prior = state_prior.to(device=hidden_states.device, dtype=torch.float32)
        bounded = torch.tanh(state_prior) * self.state_prior_scale
        if bounded.shape[1] == 1 and sequence != 1:
            bounded = bounded.expand(-1, sequence, -1)
        return bounded

    def forward(
        self,
        hidden_states: torch.Tensor,
        state_prior: torch.Tensor | None = None,
    ) -> RouterOutput:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden].")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, "
                f"got {hidden_states.shape[-1]}."
            )
        if not hidden_states.is_floating_point():
            raise TypeError("hidden_states must be a floating-point tensor.")

        # Router softmax stays in float32 under AMP for stable top-k decisions.
        token_logits = F.linear(
            hidden_states.float(),
            self.token_router.weight.float(),
            self.token_router.bias.float(),
        )
        state_bias = self._bounded_state_bias(hidden_states, state_prior)
        routing_logits = token_logits + state_bias
        probabilities = F.softmax(routing_logits, dim=-1)

        top_probabilities, expert_indices = torch.topk(
            probabilities,
            k=self.top_k,
            dim=-1,
        )
        expert_weights = top_probabilities / top_probabilities.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(torch.finfo(top_probabilities.dtype).tiny)

        assignments = F.one_hot(
            expert_indices,
            num_classes=self.num_experts,
        ).to(probabilities.dtype)
        expert_counts = assignments.sum(dim=(0, 1, 2)).to(torch.long)

        importance = probabilities.mean(dim=(0, 1))
        load = assignments.mean(dim=(0, 1, 2))
        load_balancing_loss = self.num_experts * torch.sum(importance * load)
        router_z_loss = torch.logsumexp(routing_logits, dim=-1).square().mean()
        entropy = -(
            probabilities
            * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
        ).sum(dim=-1).mean()

        return RouterOutput(
            token_logits=token_logits,
            state_bias=state_bias,
            routing_logits=routing_logits,
            probabilities=probabilities,
            expert_indices=expert_indices,
            expert_weights=expert_weights,
            expert_counts=expert_counts,
            load_balancing_loss=load_balancing_loss,
            router_z_loss=router_z_loss,
            entropy=entropy,
        )
