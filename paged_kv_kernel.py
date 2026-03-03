import mlx.core as mx
import time

def get_update_kernel(batch_size, num_kv_heads, head_dim, block_size):
    source = f"""
        uint b = thread_position_in_grid.x;
        uint h = thread_position_in_grid.y;
        uint d = thread_position_in_grid.z;
        
        if (b >= {batch_size} || h >= {num_kv_heads} || d >= {head_dim}) return;
        
        uint seq_pos = offsets[b];
        uint block_idx = seq_pos / {block_size};
        uint off_in_block = seq_pos % {block_size};
        
        uint physical_block = block_tables[b]; 
        
        size_t k_new_idx = b * ({num_kv_heads} * {head_dim}) + h * {head_dim} + d;
        bfloat16_t k_val = k_new[k_new_idx];
        bfloat16_t v_val = v_new[k_new_idx];
        
        size_t pool_idx = physical_block * ({num_kv_heads} * {block_size} * {head_dim}) 
                        + h * ({block_size} * {head_dim}) 
                        + off_in_block * {head_dim} 
                        + d;
                        
        device bfloat16_t* k_pool_mut = (device bfloat16_t*)k_pool;
        device bfloat16_t* v_pool_mut = (device bfloat16_t*)v_pool;
        
        k_pool_mut[pool_idx] = k_val;
        v_pool_mut[pool_idx] = v_val;
        
        // We still need to return something to satisfy MLX that work was done.
        // Let's just output a dummy value.
        k_pool_out[0] = 0;
        v_pool_out[0] = 0;
    """
    return mx.fast.metal_kernel(
        name="update_paged_kv",
        input_names=["k_new", "v_new", "block_tables", "offsets", "k_pool", "v_pool"],
        output_names=["k_pool_out", "v_pool_out"], 
        source=source,
    )

def test_kernel():
    num_blocks = 2048
    H = 8
    block_size = 128
    D = 128
    B = 32
    
    k_pool = mx.zeros((num_blocks, H, block_size, D), dtype=mx.bfloat16)
    v_pool = mx.zeros((num_blocks, H, block_size, D), dtype=mx.bfloat16)
    
    physical_blocks = mx.random.randint(0, num_blocks, (B,), dtype=mx.int32)
    offsets = mx.random.randint(0, block_size, (B,), dtype=mx.int32)
    
    k_new = mx.random.uniform(shape=(B, H, 1, D)).astype(mx.bfloat16)
    v_new = mx.random.uniform(shape=(B, H, 1, D)).astype(mx.bfloat16)
    
    mx.eval(k_pool, v_pool, physical_blocks, offsets, k_new, v_new)
    
    kernel = get_update_kernel(B, H, D, block_size)
    
    k_out, v_out = kernel(
        inputs=[k_new, v_new, physical_blocks, offsets, k_pool, v_pool],
        grid=(B, H, D),
        threadgroup=(1, 1, D),
        output_shapes=[(1,), (1,)],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
    )
    mx.eval(k_out, v_out, k_pool, v_pool)
    
    t0 = time.perf_counter()
    for _ in range(100):
        k_out, v_out = kernel(
            inputs=[k_new, v_new, physical_blocks, offsets, k_pool, v_pool],
            grid=(B, H, D),
            threadgroup=(1, 1, D),
            output_shapes=[(1,), (1,)],
            output_dtypes=[mx.bfloat16, mx.bfloat16],
        )
        mx.eval(k_out, v_out)
        
    t1 = time.perf_counter()
    print(f"Custom kernel: {(t1 - t0) * 10} ms per call")

if __name__ == "__main__":
    test_kernel()
