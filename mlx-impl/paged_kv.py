import mlx.core as mx

_kernels = {}

def update_paged_kv(k_new, v_new, block_tables, offsets, k_pool, v_pool, block_size):
    B, H, S_new, D = k_new.shape
    max_blocks = block_tables.shape[1]
    dtype = k_new.dtype
    key = f"update_{B}_{H}_{S_new}_{D}_{block_size}_{max_blocks}_{dtype}"
    if key not in _kernels:
        src = f"""
            uint batch_size = {B};
            uint num_kv_heads = {H};
            uint head_dim = {D};
            uint block_size = {block_size};
            uint S_new = {S_new};
            uint max_blocks = {max_blocks};

            uint b = thread_position_in_grid.x;
            uint h = thread_position_in_grid.y;
            uint s = thread_position_in_grid.z / head_dim;
            uint d = thread_position_in_grid.z % head_dim;

            if (b >= batch_size || h >= num_kv_heads || s >= S_new || d >= head_dim) return;

            uint seq_pos = offsets[b] + s;
            uint block_in_seq = seq_pos / block_size;
            uint off_in_block = seq_pos % block_size;

            uint physical_block = block_tables[b * max_blocks + block_in_seq]; 

            size_t k_new_idx = (size_t)b * ((size_t)num_kv_heads * S_new * head_dim) 
                             + (size_t)h * ((size_t)S_new * head_dim) 
                             + (size_t)s * head_dim 
                             + d;
            
            T k_val = k_new[k_new_idx];
            T v_val = v_new[k_new_idx];

            size_t pool_idx = (size_t)physical_block * ((size_t)num_kv_heads * block_size * head_dim) 
                            + (size_t)h * ((size_t)block_size * head_dim) 
                            + (size_t)off_in_block * head_dim 
                            + d;

            device T* k_pool_mut = (device T*)k_pool;
            device T* v_pool_mut = (device T*)v_pool;

            k_pool_mut[pool_idx] = k_val;
            v_pool_mut[pool_idx] = v_val;

            k_pool_out[0] = 0;
            v_pool_out[0] = 0;
        """
        _kernels[key] = mx.fast.metal_kernel(
            name="update_paged_kv",
            input_names=["k_new", "v_new", "block_tables", "offsets", "k_pool", "v_pool"],
            output_names=["k_pool_out", "v_pool_out"],
            source=src,
        )

    k_dummy, v_dummy = _kernels[key](
        inputs=[k_new, v_new, block_tables, offsets, k_pool, v_pool],
        template=[("T", dtype)],
        grid=(B, H, S_new * D),
        threadgroup=(1, 1, min(D, 1024)),
        output_shapes=[(1,), (1,)],
        output_dtypes=[dtype, dtype],
    )
    return k_dummy, v_dummy


def gather_paged_kv(block_tables, offsets, k_pool, v_pool, max_seq_len, max_blocks, block_size, B, H, D):
    dtype = k_pool.dtype
    key = f"gather_{B}_{H}_{D}_{max_seq_len}_{max_blocks}_{block_size}_{dtype}"
    if key not in _kernels:
        src = f"""
            uint batch_size = {B};
            uint num_kv_heads = {H};
            uint head_dim = {D};
            uint block_size = {block_size};
            uint max_blocks = {max_blocks};

            uint b = thread_position_in_grid.x;
            uint h = thread_position_in_grid.y;
            uint s = thread_position_in_grid.z / head_dim;
            uint d = thread_position_in_grid.z % head_dim;

            if (b >= batch_size || h >= num_kv_heads || s >= max_seq_len || d >= head_dim) return;

            uint seq_len = offsets[b];

            size_t out_idx = (size_t)b * ((size_t)num_kv_heads * max_seq_len * head_dim) 
                           + (size_t)h * ((size_t)max_seq_len * head_dim) 
                           + (size_t)s * head_dim 
                           + d;

            if (s >= seq_len) {{
                k_out[out_idx] = 0;
                v_out[out_idx] = 0;
                return;
            }}

            uint block_idx = s / block_size;
            uint off_in_block = s % block_size;

            uint physical_block = block_tables[b * max_blocks + block_idx];

            size_t pool_idx = (size_t)physical_block * ((size_t)num_kv_heads * block_size * head_dim) 
                            + (size_t)h * ((size_t)block_size * head_dim) 
                            + (size_t)off_in_block * head_dim 
                            + d;

            k_out[out_idx] = k_pool[pool_idx];
            v_out[out_idx] = v_pool[pool_idx];
        """
        _kernels[key] = mx.fast.metal_kernel(
            name="gather_paged_kv",
            input_names=["block_tables", "offsets", "k_pool", "v_pool"],
            output_names=["k_out", "v_out"],
            source=src,
        )

    k_out, v_out = _kernels[key](
        inputs=[block_tables, offsets, k_pool, v_pool],
        template=[("T", dtype), ("max_seq_len", max_seq_len)],
        grid=(B, H, max_seq_len * D),
        threadgroup=(1, 1, min(D, 1024)),
        output_shapes=[(B, H, max_seq_len, D), (B, H, max_seq_len, D)],
        output_dtypes=[dtype, dtype],
    )
    return k_out, v_out
