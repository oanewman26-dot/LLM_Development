"""Sparse SwiGLU expert dispatch for AuroraLM.

Capacity is enforced independently per sample and in causal token order.  A
future token can therefore never displace an earlier token from an expert.  If
all of a token's selected assignments overflow, its top-1 route is restored as
an explicit fallback.  This may exceed the nominal capacity, but it ensures
that the feed-forward residual is never silently replaced with zeros.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from feetforward import SwiGLU
from moe_router import MoERouter, RouterOutput


@dataclass(frozen=True)
class ExpertDispatchOutput:
    """Combined expert result and capacity telemetry."""

    hidden_states: torch.Tensor
    normalized_weights: torch.Tensor
    active_assignments: torch.Tensor
    expert_counts: torch.Tensor
    capacity: int
    dropped_assignments: int
    overflow_tokens: int


@dataclass(frozen=True)
class SparseMoEOutput:
    """Complete output from routing and expert execution."""

    hidden_states: torch.Tensor
    router: RouterOutput
    dispatch: ExpertDispatchOutput


class ExpertPool(nn.Module):
    """Own and execute a per-layer pool of SwiGLU experts."""

    def __init__(self, config: Any) -> None:
        super().__init__()

        num_experts = int(getattr(config, "num_experts", 0))
        capacity_factor = float(getattr(config, "expert_capacity_factor", 1.0))
        if num_experts < 1:
            raise ValueError("ExpertPool requires config.num_experts >= 1.")
        if capacity_factor <= 0.0 or not math.isfinite(capacity_factor):
            raise ValueError("expert_capacity_factor must be positive and finite.")

        self.hidden_size = int(config.hidden_size)
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.context_length = int(getattr(config, "context_length", 0))
        if self.context_length < 1:
            raise ValueError("ExpertPool requires config.context_length >= 1.")
        self.experts = nn.ModuleList(
            SwiGLU(config) for _ in range(self.num_experts)
        )

    def copy_dense_weights_(self, dense_ffn: SwiGLU) -> "ExpertPool":
        """Clone one dense SwiGLU into every expert for function-preserving init."""
        dense_state = dense_ffn.state_dict()
        for expert in self.experts:
            expert.load_state_dict(dense_state, strict=True)
        return self

    def _validate_routes(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> None:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden].")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, "
                f"got {hidden_states.shape[-1]}."
            )
        if hidden_states.shape[0] < 1 or hidden_states.shape[1] < 1:
            raise ValueError("ExpertPool cannot dispatch an empty token batch.")
        if hidden_states.shape[1] > self.context_length:
            raise ValueError(
                f"Sequence length {hidden_states.shape[1]} exceeds configured "
                f"context length {self.context_length}."
            )
        if expert_indices.ndim != 3:
            raise ValueError("expert_indices must have shape [batch, sequence, top_k].")
        if expert_indices.shape != expert_weights.shape:
            raise ValueError("expert_indices and expert_weights shapes must match.")
        if expert_indices.shape[:2] != hidden_states.shape[:2]:
            raise ValueError("Route batch/sequence dimensions must match hidden_states.")
        if expert_indices.shape[-1] < 1:
            raise ValueError("At least one expert route is required per token.")
        if expert_indices.dtype != torch.long:
            raise TypeError("expert_indices must use torch.long indices.")
        if not expert_weights.is_floating_point():
            raise TypeError("expert_weights must be floating point.")
        if not torch.isfinite(expert_weights).all():
            raise ValueError("expert_weights must be finite.")
        if torch.any(expert_weights < 0.0):
            raise ValueError("expert_weights cannot be negative.")
        if torch.any(expert_weights.sum(dim=-1) <= 0.0):
            raise ValueError("Every token must have positive total expert weight.")
        if torch.any(expert_indices < 0) or torch.any(
            expert_indices >= self.num_experts
        ):
            raise ValueError("expert_indices contain an out-of-range expert.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> ExpertDispatchOutput:
        self._validate_routes(hidden_states, expert_indices, expert_weights)

        batch, sequence, hidden = hidden_states.shape
        top_k = expert_indices.shape[-1]
        token_count = batch * sequence
        assignment_count = token_count * top_k
        capacity = max(
            1,
            math.ceil(
                self.capacity_factor
                * self.context_length
                * top_k
                / self.num_experts
            ),
        )

        flat_hidden = hidden_states.reshape(token_count, hidden)
        flat_indices = expert_indices.reshape(token_count, top_k)
        flat_weights = expert_weights.reshape(token_count, top_k)
        active_routes = torch.zeros_like(expert_indices, dtype=torch.bool)

        # ``nonzero`` is row-major here, so retaining the first assignments
        # makes capacity causal: later positions never change earlier output.
        # Samples are isolated so routing in one batch item cannot affect
        # another item.
        for batch_index in range(batch):
            sample_indices = expert_indices[batch_index]
            for expert_index in range(self.num_experts):
                positions = torch.nonzero(
                    sample_indices == expert_index,
                    as_tuple=False,
                )
                if positions.numel() == 0:
                    continue
                positions = positions[:capacity]
                active_routes[
                    batch_index,
                    positions[:, 0],
                    positions[:, 1],
                ] = True

        # Restore top-1 for tokens whose complete top-k set overflowed.
        unserved = ~active_routes.any(dim=-1)
        overflow_tokens = int(unserved.sum().item())
        if overflow_tokens:
            positions = torch.nonzero(unserved, as_tuple=False)
            active_routes[positions[:, 0], positions[:, 1], 0] = True

        active = active_routes.reshape(token_count, top_k)
        kept_weights = flat_weights * active.to(flat_weights.dtype)
        normalized_weights = kept_weights / kept_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(torch.finfo(kept_weights.dtype).tiny)

        combined = flat_hidden.new_zeros((token_count, hidden))
        expert_counts = torch.zeros(
            self.num_experts,
            dtype=torch.long,
            device=hidden_states.device,
        )
        for expert_index, expert in enumerate(self.experts):
            positions = torch.nonzero(
                active & (flat_indices == expert_index),
                as_tuple=False,
            )
            if positions.numel() == 0:
                continue
            token_positions = positions[:, 0]
            gates = normalized_weights[
                token_positions,
                positions[:, 1],
            ].to(hidden_states.dtype)
            expert_hidden = expert(flat_hidden.index_select(0, token_positions))
            combined = combined.index_add(
                0,
                token_positions,
                expert_hidden * gates.unsqueeze(-1),
            )
            expert_counts[expert_index] = positions.shape[0]

        dropped_assignments = assignment_count - int(active.sum().item())
        return ExpertDispatchOutput(
            hidden_states=combined.reshape(batch, sequence, hidden),
            normalized_weights=normalized_weights.reshape(
                batch,
                sequence,
                top_k,
            ),
            active_assignments=active.reshape(batch, sequence, top_k),
            expert_counts=expert_counts,
            capacity=capacity,
            dropped_assignments=dropped_assignments,
            overflow_tokens=overflow_tokens,
        )


class SparseMoE(nn.Module):
    """Token router plus the layer-local SwiGLU expert pool."""

    def __init__(
        self,
        config: Any,
        *,
        state_prior_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.router = MoERouter(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.top_k_experts,
            state_prior_scale=state_prior_scale,
            init_std=config.init_std,
        )
        self.expert_pool = ExpertPool(config)

    def copy_dense_weights_(self, dense_ffn: SwiGLU) -> "SparseMoE":
        self.expert_pool.copy_dense_weights_(dense_ffn)
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        state_prior: torch.Tensor | None = None,
    ) -> SparseMoEOutput:
        router_output = self.router(hidden_states, state_prior)
        dispatch_output = self.expert_pool(
            hidden_states,
            router_output.expert_indices,
            router_output.expert_weights,
        )
        return SparseMoEOutput(
            hidden_states=dispatch_output.hidden_states,
            router=router_output,
            dispatch=dispatch_output,
        )
