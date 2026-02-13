"""Simple tests without loading full weights."""

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
from kvcache import KVCache
from model import OLMoE, rope_freqs


def test_model_creation():
    """Test that model can be created."""
    print("[TEST] Model creation")
    model = OLMoE(
        vocab_size=50304,
        dim=2048,
        num_layers=2,  # Use fewer layers for speed
        num_heads=16,
        num_kv_heads=16,
        num_experts=8,  # Use fewer experts for speed
        active_experts=2,
        intermediate_size=1024,
        max_seq_len=2048,
        rngs=nnx.Rngs(0),
    )
    print("  PASSED")
    return model


def test_forward_no_cache():
    """Test forward pass without cache."""
    print("\n[TEST] Forward pass without cache")
    model = OLMoE(
        vocab_size=100,
        dim=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        num_experts=4,
        active_experts=2,
        intermediate_size=256,
        max_seq_len=64,
        rngs=nnx.Rngs(42),
    )

    tokens = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)  # (1, 5)
    logits, cache = model(tokens, cache=None, cur_pos=0)

    assert logits.shape == (1, 5, 100), f"Expected shape (1, 5, 100), got {logits.shape}"
    assert cache is None, "Cache should be None when not provided"
    print(f"  Output shape: {logits.shape}")
    print("  PASSED")


def test_forward_with_cache():
    """Test forward pass with KV cache."""
    print("\n[TEST] Forward pass with cache")
    model = OLMoE(
        vocab_size=100,
        dim=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        num_experts=4,
        active_experts=2,
        intermediate_size=256,
        max_seq_len=64,
        rngs=nnx.Rngs(42),
    )

    # Create cache
    cache = KVCache.new(
        layers=2,
        batch=1,
        max_seq_len=64,
        kv_heads=4,
        head_dim=128 // 4,
    )

    # Prefill with prompt
    prompt = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)  # (1, 5)
    logits, cache = model(prompt, cache=cache, cur_pos=0)

    assert logits.shape == (1, 5, 100), f"Expected shape (1, 5, 100), got {logits.shape}"
    assert cache is not None, "Cache should be returned"
    print(f"  Prefill output shape: {logits.shape}")

    # Generate next token
    next_token = jnp.array([[6]], dtype=jnp.int32)  # (1, 1)
    logits, cache = model(next_token, cache=cache, cur_pos=5)

    assert logits.shape == (1, 1, 100), f"Expected shape (1, 1, 100), got {logits.shape}"
    print(f"  Generation output shape: {logits.shape}")
    print("  PASSED")


def test_attention_shapes():
    """Test attention layer shapes."""
    print("\n[TEST] Attention layer shapes")
    from model import Attention

    attn = Attention(dim=128, num_heads=4, num_kv_heads=4, rngs=nnx.Rngs(0))

    B, S = 2, 10
    x = jnp.zeros((B, S, 128), dtype=jnp.bfloat16)
    head_dim = 128 // 4
    cos, sin = rope_freqs(head_dim, S)
    mask = jnp.tril(jnp.ones((S, S), dtype=bool))[None, None, :, :]

    out, cache = attn(x, cos, sin, mask, layer_idx=0, cache=None, cur_pos=0)

    assert out.shape == (B, S, 128), f"Expected shape (2, 10, 128), got {out.shape}"
    assert cache is None, "Cache should be None"
    print(f"  Output shape: {out.shape}")
    print("  PASSED")


def test_moe_shapes():
    """Test MoE layer shapes."""
    print("\n[TEST] MoE layer shapes")
    from model import MoEMLP

    moe = MoEMLP(
        num_experts=8,
        active_experts=2,
        dim=128,
        intermediate_size=256,
        rngs=nnx.Rngs(0),
    )

    x = jnp.zeros((2, 10, 128), dtype=jnp.bfloat16)  # (B, S, D)
    out = moe(x)

    assert out.shape == (2, 10, 128), f"Expected shape (2, 10, 128), got {out.shape}"
    print(f"  Output shape: {out.shape}")
    print("  PASSED")


def test_batch_size_gt_1():
    """Test with batch size > 1."""
    print("\n[TEST] Batch size > 1")

    model = OLMoE(
        vocab_size=100,
        dim=64,
        num_layers=1,
        num_heads=4,
        num_kv_heads=4,
        num_experts=4,
        active_experts=2,
        intermediate_size=128,
        max_seq_len=32,
        rngs=nnx.Rngs(42),
    )

    # Create batched tokens
    batch_tokens = jnp.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ],
        dtype=jnp.int32,
    )  # (2, 4)

    logits, _ = model(batch_tokens, cache=None, cur_pos=0)

    assert logits.shape == (2, 4, 100), f"Expected shape (2, 4, 100), got {logits.shape}"
    print(f"  Batch output shape: {logits.shape}")
    print("  PASSED")


def test_batch_with_cache():
    """Test batch inference with KV cache."""
    print("\n[TEST] Batch inference with cache")

    model = OLMoE(
        vocab_size=100,
        dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        num_experts=4,
        active_experts=2,
        intermediate_size=128,
        max_seq_len=32,
        rngs=nnx.Rngs(42),
    )

    # Create cache for batch of 2
    cache = KVCache.new(
        layers=2,
        batch=2,
        max_seq_len=32,
        kv_heads=4,
        head_dim=64 // 4,
    )

    # Prefill with batched prompts
    prompts = jnp.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=jnp.int32,
    )  # (2, 3)

    logits, cache = model(prompts, cache=cache, cur_pos=0)

    assert logits.shape == (2, 3, 100), f"Expected shape (2, 3, 100), got {logits.shape}"
    print(f"  Prefill output shape: {logits.shape}")

    # Generate next tokens
    next_tokens = jnp.array([[7], [8]], dtype=jnp.int32)  # (2, 1)
    logits, cache = model(next_tokens, cache=cache, cur_pos=3)

    assert logits.shape == (2, 1, 100), f"Expected shape (2, 1, 100), got {logits.shape}"
    print(f"  Generation output shape: {logits.shape}")
    print("  PASSED")


if __name__ == "__main__":
    tests = [
        test_model_creation,
        test_forward_no_cache,
        test_forward_with_cache,
        test_attention_shapes,
        test_moe_shapes,
        test_batch_size_gt_1,
        test_batch_with_cache,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        exit(1)
