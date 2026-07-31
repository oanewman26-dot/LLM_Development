import unittest
from types import SimpleNamespace

import torch

from experts import ExpertPool, SparseMoE
from feetforward import SwiGLU


def tiny_moe_config(**overrides):
    values = {
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_experts": 4,
        "top_k_experts": 2,
        "expert_capacity_factor": 1.0,
        "init_std": 0.02,
        "context_length": 6,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestExpertPool(unittest.TestCase):
    def test_dispatch_preserves_shape_and_normalizes_retained_routes(self):
        torch.manual_seed(21)
        config = tiny_moe_config()
        pool = ExpertPool(config)
        hidden = torch.randn(2, 5, config.hidden_size)
        indices = torch.randint(0, config.num_experts, (2, 5, 2))
        weights = torch.rand(2, 5, 2)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        output = pool(hidden, indices, weights)

        self.assertEqual(output.hidden_states.shape, hidden.shape)
        self.assertEqual(output.expert_counts.shape, (config.num_experts,))
        self.assertTrue(torch.isfinite(output.hidden_states).all())
        self.assertTrue(
            torch.all(output.active_assignments.any(dim=-1))
        )
        self.assertTrue(torch.allclose(
            output.normalized_weights.sum(dim=-1),
            torch.ones(2, 5),
        ))

    def test_capacity_overflow_uses_explicit_top_one_fallback(self):
        torch.manual_seed(22)
        config = tiny_moe_config(expert_capacity_factor=0.25)
        pool = ExpertPool(config)
        hidden = torch.randn(1, 6, config.hidden_size)
        indices = torch.tensor([[[0, 1]] * 6], dtype=torch.long)
        weights = torch.tensor([[[0.75, 0.25]] * 6])

        output = pool(hidden, indices, weights)

        self.assertEqual(output.capacity, 1)
        self.assertGreater(output.overflow_tokens, 0)
        self.assertGreater(output.dropped_assignments, 0)
        self.assertTrue(torch.all(output.active_assignments.any(dim=-1)))
        self.assertTrue(torch.allclose(
            output.normalized_weights.sum(dim=-1),
            torch.ones(1, 6),
        ))
        self.assertGreater(int(output.expert_counts[0]), output.capacity)

    def test_cloned_experts_preserve_dense_function_even_with_overflow(self):
        torch.manual_seed(23)
        config = tiny_moe_config(expert_capacity_factor=0.25)
        dense = SwiGLU(config)
        pool = ExpertPool(config).copy_dense_weights_(dense)
        hidden = torch.randn(2, 5, config.hidden_size)
        indices = torch.tensor([[[0, 1]] * 5, [[0, 1]] * 5])
        weights = torch.rand(2, 5, 2)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        expected = dense(hidden)
        actual = pool(hidden, indices, weights).hidden_states

        self.assertTrue(torch.allclose(expected, actual, atol=1e-6))

    def test_sparse_moe_backpropagates_to_router_and_active_experts(self):
        torch.manual_seed(24)
        config = tiny_moe_config()
        moe = SparseMoE(config)
        hidden = torch.randn(
            2,
            6,
            config.hidden_size,
            requires_grad=True,
        )

        output = moe(hidden)
        loss = (
            output.hidden_states.square().mean()
            + 0.1 * output.router.load_balancing_loss
            + 0.01 * output.router.router_z_loss
        )
        loss.backward()

        self.assertIsNotNone(hidden.grad)
        self.assertGreater(float(hidden.grad.abs().sum()), 0.0)
        self.assertIsNotNone(moe.router.token_router.weight.grad)
        self.assertGreater(
            float(moe.router.token_router.weight.grad.abs().sum()),
            0.0,
        )
        expert_gradients = [
            parameter.grad
            for expert in moe.expert_pool.experts
            for parameter in expert.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(expert_gradients)
        self.assertGreater(
            sum(float(gradient.abs().sum()) for gradient in expert_gradients),
            0.0,
        )

    def test_invalid_dispatch_contracts_are_rejected(self):
        config = tiny_moe_config()
        pool = ExpertPool(config)
        hidden = torch.randn(2, 3, config.hidden_size)
        weights = torch.full((2, 3, 2), 0.5)

        with self.assertRaises(TypeError):
            pool(hidden, torch.zeros(2, 3, 2), weights)
        with self.assertRaises(ValueError):
            pool(
                hidden,
                torch.full((2, 3, 2), config.num_experts, dtype=torch.long),
                weights,
            )
        with self.assertRaises(ValueError):
            ExpertPool(tiny_moe_config(expert_capacity_factor=0.0))


if __name__ == "__main__":
    unittest.main()
