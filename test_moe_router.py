import unittest

import torch

from moe_router import MoERouter


class TestMoERouter(unittest.TestCase):
    def test_shapes_and_normalized_top_k_weights(self):
        torch.manual_seed(11)
        router = MoERouter(8, 4, top_k=2)
        hidden = torch.randn(3, 5, 8)

        output = router(hidden)

        self.assertEqual(output.routing_logits.shape, (3, 5, 4))
        self.assertEqual(output.probabilities.shape, (3, 5, 4))
        self.assertEqual(output.expert_indices.shape, (3, 5, 2))
        self.assertEqual(output.expert_weights.shape, (3, 5, 2))
        self.assertEqual(output.expert_counts.shape, (4,))
        self.assertEqual(int(output.expert_counts.sum()), 3 * 5 * 2)
        self.assertTrue(
            torch.allclose(
                output.expert_weights.sum(dim=-1),
                torch.ones(3, 5),
            )
        )
        self.assertTrue(torch.isfinite(output.load_balancing_loss))
        self.assertTrue(torch.isfinite(output.router_z_loss))
        self.assertTrue(torch.isfinite(output.entropy))

    def test_zero_scale_keeps_state_prior_logit_neutral(self):
        torch.manual_seed(12)
        router = MoERouter(6, 3, top_k=2, state_prior_scale=0.0)
        hidden = torch.randn(2, 4, 6)
        prior = torch.randn(2, 3) * 100.0

        without_state = router(hidden)
        with_state = router(hidden, prior)

        self.assertTrue(torch.equal(with_state.state_bias, torch.zeros_like(
            with_state.state_bias
        )))
        self.assertTrue(torch.equal(
            with_state.routing_logits,
            without_state.routing_logits,
        ))
        self.assertTrue(torch.equal(
            with_state.expert_indices,
            without_state.expert_indices,
        ))

    def test_state_prior_is_bounded_and_cannot_override_large_token_gap(self):
        router = MoERouter(
            2,
            2,
            top_k=1,
            state_prior_scale=0.25,
        )
        with torch.no_grad():
            router.token_router.weight.zero_()
            router.token_router.bias.copy_(torch.tensor([2.0, 0.0]))

        hidden = torch.zeros(1, 3, 2)
        prior = torch.tensor([[-1000.0, 1000.0]])
        output = router(hidden, prior)

        self.assertLessEqual(float(output.state_bias.abs().max()), 0.25)
        self.assertTrue(torch.equal(
            output.expert_indices,
            torch.zeros_like(output.expert_indices),
        ))

    def test_collapsed_routes_have_larger_balance_loss(self):
        router = MoERouter(2, 2, top_k=1)
        with torch.no_grad():
            router.token_router.weight.copy_(torch.eye(2))
            router.token_router.bias.zero_()

        balanced = torch.tensor([[
            [8.0, -8.0],
            [-8.0, 8.0],
            [8.0, -8.0],
            [-8.0, 8.0],
        ]])
        collapsed = torch.tensor([[[8.0, -8.0]] * 4])

        balanced_loss = router(balanced).load_balancing_loss
        collapsed_loss = router(collapsed).load_balancing_loss

        self.assertGreater(
            float(collapsed_loss.detach()),
            float(balanced_loss.detach()),
        )

    def test_gradients_reach_hidden_states_and_token_router(self):
        torch.manual_seed(13)
        router = MoERouter(5, 4, top_k=2)
        hidden = torch.randn(2, 3, 5, requires_grad=True)

        output = router(hidden)
        loss = (
            output.expert_weights.square().mean()
            + output.load_balancing_loss
            + 0.01 * output.router_z_loss
        )
        loss.backward()

        self.assertIsNotNone(hidden.grad)
        self.assertGreater(float(hidden.grad.abs().sum()), 0.0)
        self.assertIsNotNone(router.token_router.weight.grad)
        self.assertGreater(
            float(router.token_router.weight.grad.abs().sum()),
            0.0,
        )

    def test_invalid_router_contracts_are_rejected(self):
        with self.assertRaises(ValueError):
            MoERouter(8, 0)
        with self.assertRaises(ValueError):
            MoERouter(8, 4, top_k=5)

        router = MoERouter(8, 4)
        with self.assertRaises(ValueError):
            router(torch.randn(2, 8))
        with self.assertRaises(ValueError):
            router(torch.randn(2, 3, 8), torch.randn(3, 4))


if __name__ == "__main__":
    unittest.main()
