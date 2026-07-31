import unittest

import torch

from config import AuroraLMConfig
from NN import AuroraLM
from transformer_block import TransformerBlock


def tiny_config(*, num_experts):
    return AuroraLMConfig(
        name=f"moe_integration_{num_experts}",
        vocab_size=64,
        context_length=8,
        hidden_size=16,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=24,
        state_size=4,
        state_hidden_dim=8,
        num_experts=num_experts,
        top_k_experts=2,
        expert_capacity_factor=0.5,
    )


class TestMoEIntegration(unittest.TestCase):
    def test_dense_model_keeps_logits_only_default_and_empty_telemetry(self):
        torch.manual_seed(30)
        config = tiny_config(num_experts=0)
        model = AuroraLM(config).eval()
        tokens = torch.randint(0, config.vocab_size, (2, 6))

        with torch.no_grad():
            default_logits = model(tokens)
            telemetry_logits, routing_outputs = model(
                tokens,
                return_routing=True,
            )

        self.assertTrue(torch.equal(default_logits, telemetry_logits))
        self.assertEqual(routing_outputs, ())

    def test_model_returns_per_layer_routing_telemetry_on_request(self):
        torch.manual_seed(31)
        config = tiny_config(num_experts=4)
        model = AuroraLM(config)
        tokens = torch.randint(0, config.vocab_size, (2, 6))
        state = torch.randn(2, config.source_state_dim)

        logits, routing_outputs = model(
            tokens,
            state,
            return_routing=True,
        )

        self.assertEqual(logits.shape, (2, 6, config.vocab_size))
        self.assertEqual(len(routing_outputs), config.num_layers)
        for output in routing_outputs:
            self.assertEqual(
                output.router.expert_indices.shape,
                (2, 6, config.top_k_experts),
            )
            self.assertEqual(output.dispatch.hidden_states.shape, (2, 6, 16))
            self.assertTrue(
                torch.all(output.dispatch.active_assignments.any(dim=-1))
            )

    def test_moe_preserves_zero_init_state_baseline(self):
        torch.manual_seed(32)
        config = tiny_config(num_experts=4)
        model = AuroraLM(config).eval()
        tokens = torch.randint(0, config.vocab_size, (2, 6))
        state = torch.randn(2, config.source_state_dim)

        with torch.no_grad():
            without_state = model(tokens, None)
            with_state = model(tokens, state)

        self.assertTrue(torch.equal(without_state, with_state))

    def test_moe_capacity_decisions_do_not_leak_future_tokens(self):
        torch.manual_seed(33)
        config = tiny_config(num_experts=4)
        model = AuroraLM(config).eval()
        tokens = torch.randint(0, config.vocab_size, (2, 8))
        state = torch.randn(2, config.source_state_dim)

        with torch.no_grad():
            full_logits = model(tokens, state)
            prefix_logits = model(tokens[:, :4], state)

        self.assertTrue(torch.allclose(
            full_logits[:, :4],
            prefix_logits,
            atol=1e-6,
        ))

    def test_cloned_experts_preserve_dense_transformer_block(self):
        torch.manual_seed(34)
        dense_config = tiny_config(num_experts=0)
        moe_config = tiny_config(num_experts=4)
        dense_block = TransformerBlock(dense_config).eval()
        moe_block = TransformerBlock(moe_config).eval()

        moe_block.attn_norm.load_state_dict(dense_block.attn_norm.state_dict())
        moe_block.ffn_norm.load_state_dict(dense_block.ffn_norm.state_dict())
        moe_block.attn.load_state_dict(dense_block.attn.state_dict())
        moe_block.attn_cond.load_state_dict(dense_block.attn_cond.state_dict())
        moe_block.ffn_cond.load_state_dict(dense_block.ffn_cond.state_dict())
        moe_block.ffn.copy_dense_weights_(dense_block.ffn)

        hidden = torch.randn(2, 6, dense_config.hidden_size)
        state_latent = torch.randn(2, dense_config.state_size)
        routing_prior = torch.randn(2, moe_config.num_experts)

        with torch.no_grad():
            dense_output = dense_block(hidden, state_latent)
            moe_output = moe_block(
                hidden,
                state_latent,
                routing_prior,
            )

        self.assertTrue(torch.allclose(dense_output, moe_output, atol=1e-6))

    def test_moe_parameter_estimate_matches_model(self):
        config = tiny_config(num_experts=4)
        model = AuroraLM(config)
        actual = sum(parameter.numel() for parameter in model.parameters())

        self.assertEqual(actual, config.estimated_params())


if __name__ == "__main__":
    unittest.main()
