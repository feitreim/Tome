"""Comparison tests: our OLMoE vs HuggingFace transformers."""

import device  # Auto-configures JAX device (CUDA/MPS/CPU)
import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import torch

MODEL_NAME = "allenai/OLMoE-1B-7B-0924"
SEQ_LEN = 32
ATOL = 1e-2  # bf16 tolerance for single-layer tests
ATOL_FULL = 1.0  # accumulated bf16 tolerance for full 16-layer model


def _get_hf_model():
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cpu")


def _get_our_model(checkpoint_path: str):
    from load_weights import load_olmoe_weights
    from model import OLMoE

    model = OLMoE(
        vocab_size=50304,
        dim=2048,
        num_layers=16,
        num_heads=16,
        num_kv_heads=16,
        num_experts=64,
        active_experts=8,
        intermediate_size=1024,
        max_seq_len=2048,
        rngs=nnx.Rngs(0),
    )
    load_olmoe_weights(model, checkpoint_path)
    return model


def _make_tokens(seq_len=SEQ_LEN):
    rng = np.random.RandomState(42)
    return rng.randint(0, 50304, size=(1, seq_len))


def _hf_position_embeddings(hf_model, x_pt, S):
    """Get (cos, sin) position embeddings from HF model's rotary_emb."""
    position_ids = torch.arange(S).unsqueeze(0)
    cos, sin = hf_model.model.rotary_emb(x_pt, position_ids)
    return cos, sin


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_rmsnorm(hf_model, our_model):
    tokens = _make_tokens()
    x_pt = hf_model.model.embed_tokens(torch.tensor(tokens))  # (1, S, D)
    x_jax = jnp.array(x_pt.detach().float().numpy(), dtype=jnp.bfloat16)  # Keep batch dim

    # HF RMSNorm
    hf_out = hf_model.model.layers[0].input_layernorm(x_pt)
    hf_np = hf_out.detach().float().numpy()

    # Our RMSNorm
    our_out = our_model.layers[0].input_norm(x_jax)
    our_np = np.array(our_out, dtype=np.float32)

    diff = np.max(np.abs(hf_np - our_np))
    print(f"  RMSNorm max diff: {diff:.6f}")
    assert diff < ATOL, f"RMSNorm diff {diff} exceeds {ATOL}"


def test_attention_layer(hf_model, our_model):
    tokens = _make_tokens()
    x_pt = hf_model.model.embed_tokens(torch.tensor(tokens))
    normed_pt = hf_model.model.layers[0].input_layernorm(x_pt)

    S = tokens.shape[1]
    causal_mask = torch.triu(torch.full((S, S), float("-inf")), diagonal=1)
    causal_mask = causal_mask[None, None, :, :]  # (1, 1, S, S)

    # HF attention — needs position_embeddings (cos, sin) tuple
    position_embeddings = _hf_position_embeddings(hf_model, normed_pt, S)
    hf_attn_out = hf_model.model.layers[0].self_attn(
        normed_pt, position_embeddings=position_embeddings, attention_mask=causal_mask
    )[0]
    hf_np = hf_attn_out.detach().float().numpy()

    # Our attention
    normed_jax = jnp.array(normed_pt.detach().float().numpy(), dtype=jnp.bfloat16)  # Keep batch dim
    from model import rope_freqs

    head_dim = 2048 // 16
    cos, sin = rope_freqs(head_dim, S)
    mask = jnp.tril(jnp.ones((S, S), dtype=bool))[None, None, :, :]
    our_attn_out, _ = our_model.layers[0].attn(normed_jax, cos, sin, mask, layer_idx=0, cache=None, cur_pos=0)
    our_np = np.array(our_attn_out, dtype=np.float32)

    diff = np.max(np.abs(hf_np - our_np))
    print(f"  Attention max diff: {diff:.6f}")
    assert diff < ATOL, f"Attention diff {diff} exceeds {ATOL}"


def test_moe_layer(hf_model, our_model):
    tokens = _make_tokens()
    x_pt = hf_model.model.embed_tokens(torch.tensor(tokens))

    S = tokens.shape[1]
    causal_mask = torch.triu(torch.full((S, S), float("-inf")), diagonal=1)[None, None, :, :]

    position_embeddings = _hf_position_embeddings(hf_model, x_pt, S)

    normed = hf_model.model.layers[0].input_layernorm(x_pt)
    attn_out = hf_model.model.layers[0].self_attn(
        normed, position_embeddings=position_embeddings, attention_mask=causal_mask
    )[0]
    residual = x_pt + attn_out
    moe_input_pt = hf_model.model.layers[0].post_attention_layernorm(residual)

    # HF MoE
    hf_moe_out = hf_model.model.layers[0].mlp(moe_input_pt)
    hf_np = hf_moe_out.detach().float().numpy()

    # Our MoE
    moe_input_jax = jnp.array(moe_input_pt.detach().float().numpy(), dtype=jnp.bfloat16)  # Keep batch dim
    our_moe_out = our_model.layers[0].moe(moe_input_jax)
    our_np = np.array(our_moe_out, dtype=np.float32)

    diff = np.max(np.abs(hf_np - our_np))
    print(f"  MoE max diff: {diff:.6f}")
    assert diff < ATOL, f"MoE diff {diff} exceeds {ATOL}"


def test_full_model_output(hf_model, our_model):
    tokens = _make_tokens()

    # HF forward
    with torch.no_grad():
        hf_out = hf_model(torch.tensor(tokens))
    hf_logits = hf_out.logits.float().numpy()

    # Our forward
    our_logits, _ = our_model(jnp.array(tokens), cache=None, cur_pos=0)
    our_logits_np = np.array(our_logits, dtype=np.float32)

    diff = np.max(np.abs(hf_logits - our_logits_np))
    print(f"  Full model max diff: {diff:.6f}")
    assert diff < ATOL_FULL, f"Full model diff {diff} exceeds {ATOL_FULL}"


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


if __name__ == "__main__":
    from load_weights import download_olmoe

    print("Downloading/locating checkpoint...")
    checkpoint_path = download_olmoe(MODEL_NAME)
    print(f"Checkpoint at: {checkpoint_path}")

    print("Loading HF model...")
    hf_model = _get_hf_model()
    hf_model.eval()

    print("Loading our model...")
    our_model = _get_our_model(checkpoint_path)

    tests = [
        ("RMSNorm", test_rmsnorm),
        ("Attention layer", test_attention_layer),
        ("MoE layer", test_moe_layer),
        ("Full model output", test_full_model_output),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n[TEST] {name}")
        try:
            fn(hf_model, our_model)
            print("  PASSED")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        exit(1)
