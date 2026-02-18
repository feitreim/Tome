"""Comparison tests: our MLX model vs HuggingFace transformers (using pre-generated reference outputs)."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

MODEL_NAME = "Nanbeige/Nanbeige4.1-3B"
SEQ_LEN = 32
ATOL = 0.1  # bf16 accumulation across multi-layer matmuls
ATOL_FULL = 2.0  # accumulated error across 32 decoder layers
TEST_DATA_DIR = Path(__file__).parent / "test_inputs"


def _get_our_model(checkpoint_path: str):
    from load_weights import load_qwen3_weights
    from model import Qwen3

    model = Qwen3(
        vocab_size=166144,
        dim=2560,
        num_layers=32,
        num_heads=20,
        num_kv_heads=4,
        head_dim=128,
        intermediate_size=10496,
        max_seq_len=262144,
        rope_theta=70000000.0,
        eps=1e-5,
        tie_word_embeddings=False,
        use_qk_norm=False,
        rope_traditional=False,
    )
    load_qwen3_weights(model, checkpoint_path)
    return model


def _check_test_data():
    required_files = [
        "tokens.npy",
        "embeddings.npy",
        "rmsnorm_layer0.npy",
        "attention_layer0.npy",
        "mlp_input_layer0.npy",
        "mlp_output_layer0.npy",
        "full_model_logits.npy",
        "metadata.json",
    ]

    if not TEST_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Test data directory not found: {TEST_DATA_DIR}\n"
            f"Run 'uv run mlx-impl/generate_test_data.py' to generate reference outputs first."
        )

    missing = [f for f in required_files if not (TEST_DATA_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing test data files: {missing}\n"
            f"Run 'uv run mlx-impl/generate_test_data.py' to generate reference outputs."
        )


def test_rmsnorm(our_model):
    embeddings = np.load(TEST_DATA_DIR / "embeddings.npy")
    hf_rmsnorm = np.load(TEST_DATA_DIR / "rmsnorm_layer0.npy")

    x_mlx = mx.array(embeddings).astype(mx.bfloat16)
    our_out = our_model.layers[0].input_layernorm(x_mlx)
    our_np = np.array(our_out.astype(mx.float32))

    diff = np.max(np.abs(hf_rmsnorm - our_np))
    print(f"  RMSNorm max diff: {diff:.6f}")
    assert diff < ATOL, f"RMSNorm diff {diff} exceeds {ATOL}"


def test_attention_layer(our_model):
    rmsnorm_out = np.load(TEST_DATA_DIR / "rmsnorm_layer0.npy")
    hf_attention = np.load(TEST_DATA_DIR / "attention_layer0.npy")

    normed_mlx = mx.array(rmsnorm_out).astype(mx.bfloat16)
    mask = "causal"
    our_attn_out, _ = our_model.layers[0].self_attn(normed_mlx, mask, layer_idx=0, cache=None)
    mx.eval(our_attn_out)
    our_np = np.array(our_attn_out.astype(mx.float32))

    diff = np.max(np.abs(hf_attention - our_np))
    print(f"  Attention max diff: {diff:.6f}")
    assert diff < ATOL, f"Attention diff {diff} exceeds {ATOL}"


def test_mlp_layer(our_model):
    mlp_input = np.load(TEST_DATA_DIR / "mlp_input_layer0.npy")
    hf_mlp = np.load(TEST_DATA_DIR / "mlp_output_layer0.npy")

    mlp_input_mlx = mx.array(mlp_input).astype(mx.bfloat16)
    our_mlp_out = our_model.layers[0].mlp(mlp_input_mlx)
    mx.eval(our_mlp_out)
    our_np = np.array(our_mlp_out.astype(mx.float32))

    diff = np.max(np.abs(hf_mlp - our_np))
    print(f"  MLP max diff: {diff:.6f}")
    assert diff < ATOL, f"MLP diff {diff} exceeds {ATOL}"


def test_full_model_output(our_model):
    tokens = np.load(TEST_DATA_DIR / "tokens.npy")
    hf_logits = np.load(TEST_DATA_DIR / "full_model_logits.npy")

    our_logits, _ = our_model(mx.array(tokens), cache=None, cur_pos=0)
    mx.eval(our_logits)
    our_logits_np = np.array(our_logits.astype(mx.float32))

    diff = np.max(np.abs(hf_logits - our_logits_np))
    print(f"  Full model max diff: {diff:.6f}")
    assert diff < ATOL_FULL, f"Full model diff {diff} exceeds {ATOL_FULL}"


if __name__ == "__main__":
    from load_weights import download_qwen3

    print("Checking test data...")
    _check_test_data()

    with open(TEST_DATA_DIR / "metadata.json") as f:
        metadata = json.load(f)
    print(f"  Found test data for {metadata['model_name']}")

    print("\nDownloading/locating checkpoint...")
    checkpoint_path = download_qwen3(MODEL_NAME)
    print(f"Checkpoint at: {checkpoint_path}")

    print("Loading our model...")
    our_model = _get_our_model(checkpoint_path)

    tests = [
        ("RMSNorm", test_rmsnorm),
        ("Attention layer", test_attention_layer),
        ("MLP layer", test_mlp_layer),
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
