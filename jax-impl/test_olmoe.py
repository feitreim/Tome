"""Comparison tests: our OLMoE vs HuggingFace transformers (using pre-generated reference outputs)."""

import json
from pathlib import Path

import device  # Auto-configures JAX device (CUDA/MPS/CPU)
import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np

MODEL_NAME = "allenai/OLMoE-1B-7B-0924"
SEQ_LEN = 32
ATOL = 1e-2  # bf16 tolerance for single-layer tests
ATOL_FULL = 1.0  # accumulated bf16 tolerance for full 16-layer model
TEST_DATA_DIR = Path("test_inputs")


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


def _check_test_data():
    """Check if test data exists, otherwise instruct user to generate it."""
    required_files = [
        "tokens.npy",
        "embeddings.npy",
        "rmsnorm_layer0.npy",
        "attention_layer0.npy",
        "moe_input_layer0.npy",
        "moe_output_layer0.npy",
        "full_model_logits.npy",
        "metadata.json",
    ]

    if not TEST_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Test data directory not found: {TEST_DATA_DIR}\n"
            f"Run 'uv run jax-impl/generate_test_data.py' to generate reference outputs first."
        )

    missing = [f for f in required_files if not (TEST_DATA_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing test data files: {missing}\n"
            f"Run 'uv run jax-impl/generate_test_data.py' to generate reference outputs."
        )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_rmsnorm(our_model):
    embeddings = np.load(TEST_DATA_DIR / "embeddings.npy")
    hf_rmsnorm = np.load(TEST_DATA_DIR / "rmsnorm_layer0.npy")

    x_jax = jnp.array(embeddings, dtype=jnp.bfloat16)
    our_out = our_model.layers[0].input_norm(x_jax)
    our_np = np.array(our_out, dtype=np.float32)

    diff = np.max(np.abs(hf_rmsnorm - our_np))
    print(f"  RMSNorm max diff: {diff:.6f}")
    assert diff < ATOL, f"RMSNorm diff {diff} exceeds {ATOL}"


def test_attention_layer(our_model):
    tokens = np.load(TEST_DATA_DIR / "tokens.npy")
    rmsnorm_out = np.load(TEST_DATA_DIR / "rmsnorm_layer0.npy")
    hf_attention = np.load(TEST_DATA_DIR / "attention_layer0.npy")

    S = tokens.shape[1]

    normed_jax = jnp.array(rmsnorm_out, dtype=jnp.bfloat16)
    from model import rope_freqs

    head_dim = 2048 // 16
    cos, sin = rope_freqs(head_dim, S)
    mask = jnp.tril(jnp.ones((S, S), dtype=bool))[None, None, :, :]
    our_attn_out, _ = our_model.layers[0].attn(normed_jax, cos, sin, mask, layer_idx=0, cache=None, cur_pos=0)
    our_np = np.array(our_attn_out, dtype=np.float32)

    diff = np.max(np.abs(hf_attention - our_np))
    print(f"  Attention max diff: {diff:.6f}")
    assert diff < ATOL, f"Attention diff {diff} exceeds {ATOL}"


def test_moe_layer(our_model):
    moe_input = np.load(TEST_DATA_DIR / "moe_input_layer0.npy")
    hf_moe = np.load(TEST_DATA_DIR / "moe_output_layer0.npy")

    moe_input_jax = jnp.array(moe_input, dtype=jnp.bfloat16)
    our_moe_out = our_model.layers[0].moe(moe_input_jax)
    our_np = np.array(our_moe_out, dtype=np.float32)

    diff = np.max(np.abs(hf_moe - our_np))
    print(f"  MoE max diff: {diff:.6f}")
    assert diff < ATOL, f"MoE diff {diff} exceeds {ATOL}"


def test_full_model_output(our_model):
    tokens = np.load(TEST_DATA_DIR / "tokens.npy")
    hf_logits = np.load(TEST_DATA_DIR / "full_model_logits.npy")

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

    print("Checking test data...")
    _check_test_data()

    with open(TEST_DATA_DIR / "metadata.json") as f:
        metadata = json.load(f)
    print(f"  Found test data for {metadata['model_name']}")

    print("\nDownloading/locating checkpoint...")
    checkpoint_path = download_olmoe(MODEL_NAME)
    print(f"Checkpoint at: {checkpoint_path}")

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
            fn(our_model)
            print("  PASSED")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        exit(1)
