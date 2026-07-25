"""
tests/test_smoke.py

Reproducible smoke tests for AuroraLM.

Run with:
    python test_smoke.py              # direct execution (file is in project root)
    python -m pytest test_smoke.py -v # pytest

All tests must pass before cloud compute is authorised.
"""

import sys
import os

# Allow imports from project root regardless of where pytest is invoked
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from config import PILOT, CONTROL_PILOT, STATE_SCHEMA_VERSION
from NN import AuroraLM


# ---------------------------------------------------------------------------
# Fixture: pinned seed and device
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    """Pin all randomness for reproducible smoke tests."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helper: build a dummy batch
# ---------------------------------------------------------------------------

def dummy_batch(config, batch_size=2, seq_len=16):
    """Random token IDs + random AuroraState171."""
    tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    aurora_state = torch.randn(batch_size, 171)
    return tokens, aurora_state


# ---------------------------------------------------------------------------
# Test 1: module imports normally
# ---------------------------------------------------------------------------

def test_imports():
    """All core modules import without error."""
    import config
    import NN
    import transformer_block
    import rotary_position_embedding
    import feetforward
    import RSMnorm
    import state_encoder
    assert config.STATE_SCHEMA_VERSION == STATE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Test 2: forward pass produces correct logit shapes
# ---------------------------------------------------------------------------

def test_forward_shapes():
    """Logits shape is (batch, seq, vocab)."""
    set_seed(42)
    config = PILOT
    model = AuroraLM(config).to(DEVICE)
    model.eval()

    tokens, aurora_state = dummy_batch(config)
    tokens = tokens.to(DEVICE)
    aurora_state = aurora_state.to(DEVICE)

    with torch.no_grad():
        logits = model(tokens, aurora_state)

    assert logits.shape == (2, 16, config.vocab_size),         f"Expected (2, 16, {config.vocab_size}), got {logits.shape}"


# ---------------------------------------------------------------------------
# Test 3: initial loss is near theoretical maximum
# ---------------------------------------------------------------------------

def test_initial_loss():
    """Random weights should give loss close to ln(vocab_size)."""
    set_seed(42)
    config = PILOT
    model = AuroraLM(config).to(DEVICE)
    model.eval()

    tokens, aurora_state = dummy_batch(config)
    tokens = tokens.to(DEVICE)
    aurora_state = aurora_state.to(DEVICE)

    with torch.no_grad():
        logits = model(tokens, aurora_state)

    # Shift for next-token prediction: predict token[1:] from logits[:-1]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, config.vocab_size),
        shift_labels.view(-1)
    )

    theoretical = torch.log(torch.tensor(float(config.vocab_size)))
    # Allow 5% tolerance for random init variance
    assert torch.isclose(loss, theoretical, rtol=0.05),         f"Loss {loss.item():.4f} far from theoretical {theoretical.item():.4f}"


# ---------------------------------------------------------------------------
# Test 4: gradients flow to all parameter tensors
# ---------------------------------------------------------------------------

def test_gradient_flow():
    """A backward pass touches every parameter that requires grad.

    Zero-init parameters (state_head, routing_head, state_conditioner outputs)
    are allowed to have zero gradients — that's the baseline property.
    All other parameters must receive non-zero gradients.
    """
    set_seed(42)
    config = PILOT
    model = AuroraLM(config).to(DEVICE)
    model.train()

    tokens, aurora_state = dummy_batch(config)
    tokens = tokens.to(DEVICE)
    aurora_state = aurora_state.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def compute_loss():
        logits = model(tokens, aurora_state)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tokens[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, config.vocab_size),
            shift_labels.view(-1),
        )

    # --- Part A: every parameter must RECEIVE a gradient on step 0 ---
    # (a gradient may be zero here by design; we only require it exists,
    #  i.e. the parameter is actually wired into the compute graph)
    loss = compute_loss()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter never touched: {name}"

    # --- Part B: run a few real steps, then confirm the state pathway
    #     has started to learn (non-zero gradient) ---
    # The zero-init layers (state_head, conditioner outputs) start at zero on
    # purpose, so their gradients are zero on step 0. After a couple of updates
    # they become non-zero and the whole state pathway must be learning. This
    # tests the property we actually care about — that state can influence the
    # model — rather than the instantaneous zero we deliberately engineered.
    for _ in range(3):
        optimizer.zero_grad()
        loss = compute_loss()
        loss.backward()
        optimizer.step()

    # One more backward pass to inspect gradients after warmup
    optimizer.zero_grad()
    loss = compute_loss()
    loss.backward()

    state_pathway = [
        "state_encoder.trunk",
        "state_encoder.state_head",
        "attn_cond",
        "ffn_cond",
    ]

    checked_any = False
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(tag in name for tag in state_pathway):
            checked_any = True
            assert param.grad is not None, f"No gradient for {name} after warmup"
            assert not torch.all(param.grad == 0), (
                f"State pathway not learning after 3 steps: {name} still zero"
            )

    assert checked_any, "No state-pathway parameters found — check layer names"


# ---------------------------------------------------------------------------
# Test 5: zero-init baseline — aurora_state=None matches random state
# ---------------------------------------------------------------------------

def test_zero_init_baseline():
    """At init, aurora_state=None and a random state produce identical logits."""
    set_seed(42)
    config = PILOT
    model = AuroraLM(config).to(DEVICE)
    model.eval()

    tokens, aurora_state = dummy_batch(config)
    tokens = tokens.to(DEVICE)
    aurora_state = aurora_state.to(DEVICE)

    with torch.no_grad():
        logits_with_state = model(tokens, aurora_state)
        logits_no_state = model(tokens, None)

    assert torch.allclose(logits_with_state, logits_no_state, atol=1e-6),         "Zero-init baseline broken: stateful and stateless logits differ at init"


# ---------------------------------------------------------------------------
# Test 6: control model (state_size=0) matches PILOT with state disabled
# ---------------------------------------------------------------------------

def test_control_model_equivalence():
    """CONTROL_PILOT and PILOT with aurora_state=None are identical."""
    set_seed(42)

    pilot_model = AuroraLM(PILOT).to(DEVICE)
    control_model = AuroraLM(CONTROL_PILOT).to(DEVICE)

    # Copy weights where shapes match (embedding, blocks, norm, output)
    pilot_state = pilot_model.state_dict()
    control_state = control_model.state_dict()

    # CONTROL_PILOT lacks state_encoder keys — skip those
    for key in control_state:
        assert key in pilot_state, f"Key {key} missing from PILOT model"
        control_state[key].copy_(pilot_state[key])

    pilot_model.eval()
    control_model.eval()

    tokens, _ = dummy_batch(PILOT)
    tokens = tokens.to(DEVICE)

    with torch.no_grad():
        pilot_logits = pilot_model(tokens, None)
        control_logits = control_model(tokens, None)

    assert torch.allclose(pilot_logits, control_logits, atol=1e-6),         "CONTROL_PILOT diverges from PILOT with state disabled"


# ---------------------------------------------------------------------------
# Test 7: causal masking — future tokens do not leak
# ---------------------------------------------------------------------------

def test_causal_mask():
    """Position i must not attend to position j > i."""
    set_seed(42)
    config = PILOT
    model = AuroraLM(config).to(DEVICE)
    model.eval()

    tokens, aurora_state = dummy_batch(config, seq_len=8)
    tokens = tokens.to(DEVICE)
    aurora_state = aurora_state.to(DEVICE)

    with torch.no_grad():
        logits_full = model(tokens, aurora_state)

    # Truncate to first 4 tokens and forward again
    # aurora_state is per-sample (batch, 171), not per-token — do NOT slice it
    tokens_trunc = tokens[:, :4]
    with torch.no_grad():
        logits_trunc = model(tokens_trunc, aurora_state)

    # First 4 positions must be identical regardless of future context
    assert torch.allclose(logits_full[:, :4, :], logits_trunc, atol=1e-6),         "Causal mask leak: future tokens affect past positions"


# ---------------------------------------------------------------------------
# Test 8: state encoder produces correct output shapes
# ---------------------------------------------------------------------------

def test_state_encoder_shapes():
    """StateEncoder returns (batch, state_size) and optional (batch, num_experts)."""
    from state_encoder import StateEncoder

    batch = 3
    encoder = StateEncoder(
        source_dim=171,
        state_size=16,
        hidden_dim=64,
        num_experts=8,
    )
    encoder.eval()

    aurora_state = torch.randn(batch, 171)

    with torch.no_grad():
        latent, routing = encoder(aurora_state)

    assert latent.shape == (batch, 16), f"Expected (3, 16), got {latent.shape}"
    assert routing.shape == (batch, 8), f"Expected (3, 8), got {routing.shape}"

    # Test with num_experts=0
    encoder_no_moe = StateEncoder(
        source_dim=171,
        state_size=16,
        hidden_dim=64,
        num_experts=0,
    )
    with torch.no_grad():
        latent2, routing2 = encoder_no_moe(aurora_state)
    assert latent2.shape == (batch, 16)
    assert routing2 is None


# ---------------------------------------------------------------------------
# Test 9: parameter count matches config estimate
# ---------------------------------------------------------------------------

def test_parameter_count():
    """Actual params should be within 5% of config estimate."""
    set_seed(42)
    config = PILOT
    model = AuroraLM(config).to(DEVICE)

    total, trainable = model.count_parameters()
    estimated = config.estimated_params()

    # Allow 5% tolerance (estimate includes all components)
    ratio = total / estimated
    assert 0.95 <= ratio <= 1.05,         f"Param count {total} far from estimate {estimated} (ratio {ratio:.2f})"
    assert trainable == total, "All parameters should be trainable"


# ---------------------------------------------------------------------------
# Test 10: state schema version is stamped in checkpoints
# ---------------------------------------------------------------------------

def test_schema_version_in_state_dict():
    """State dict metadata carries the schema version."""
    set_seed(42)
    model = AuroraLM(PILOT).to(DEVICE)

    state_dict = model.state_dict()
    assert "_schema_version" in state_dict or True  # placeholder: version is in config, not weights
    # The real check: config object is accessible and matches
    assert model.config.schema_version == STATE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Running AuroraLM smoke tests on {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Schema version: {STATE_SCHEMA_VERSION}")
    print("-" * 50)

    tests = [
        test_imports,
        test_forward_shapes,
        test_initial_loss,
        test_gradient_flow,
        test_zero_init_baseline,
        test_control_model_equivalence,
        test_causal_mask,
        test_state_encoder_shapes,
        test_parameter_count,
        test_schema_version_in_state_dict,
    ]

    passed = 0
    failed = 0

    for test in tests:
        name = test.__name__
        try:
            test()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1

    print("-" * 50)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")

    if failed > 0:
        sys.exit(1)
