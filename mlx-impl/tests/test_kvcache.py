import mlx.core as mx
import numpy as np
import sys
import os

# Ensure we can import from the parent directory (mlx-impl)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvcache import BlockAllocator, KVCache

def test_kv_cache_batched():
    print("\n[TEST] KVCache Batched (B=2)")
    num_layers = 1
    num_kv_heads = 1
    head_dim = 8
    block_size = 4
    num_blocks = 20
    
    allocator = BlockAllocator(num_blocks, block_size, num_layers, num_kv_heads, head_dim)
    cache = KVCache(num_layers, num_kv_heads, head_dim, max_seq_len=100, batch_size=2, allocator=allocator)
    
    # 1. Prefill (B=2, S=3)
    k = mx.random.uniform(shape=(2, 1, 3, 8)).astype(mx.bfloat16)
    v = mx.random.uniform(shape=(2, 1, 3, 8)).astype(mx.bfloat16)
    
    k_out, v_out = cache.update(k, v, layer_idx=0)
    cache.advance(3)
    
    assert k_out.shape == (2, 1, 3, 8)
    assert np.allclose(np.array(k_out.astype(mx.float32)), np.array(k.astype(mx.float32)), atol=1e-5)
    
    # 2. Decode (B=2, S=1)
    k2 = mx.random.uniform(shape=(2, 1, 1, 8)).astype(mx.bfloat16)
    v2 = mx.random.uniform(shape=(2, 1, 1, 8)).astype(mx.bfloat16)
    
    k_out, v_out = cache.update(k2, v2, layer_idx=0)
    cache.advance(1)
    
    assert k_out.shape == (2, 1, 4, 8)
    expected_k = mx.concatenate([k, k2], axis=2)
    assert np.allclose(np.array(k_out.astype(mx.float32)), np.array(expected_k.astype(mx.float32)), atol=1e-5)
    print("  KVCache Batched PASSED")

if __name__ == "__main__":
    test_kv_cache_batched()
