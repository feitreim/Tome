import mlx.core as mx
import numpy as np

_kernels = {}

def update_paged_kv(k_new, v_new, block_tables, offsets, k_pool, v_pool, block_size, lengths=None):
    B, H, S_new, D = k_new.shape
    
    # Identify all blocks to be updated across the whole batch
    k_pool_out = k_pool
    v_pool_out = v_pool
    
    block_tables_np = np.array(block_tables)
    offsets_np = np.array(offsets)
    lengths_np = np.array(lengths) if lengths is not None else np.full((B,), S_new)
    
    # Collect all updates for this batch
    indices = []
    k_pieces = []
    v_pieces = []
    
    for b in range(B):
        offset = int(offsets_np[b])
        num_toks_to_write = int(lengths_np[b])
        
        num_blocks_to_update = (num_toks_to_write + block_size - 1) // block_size
        for i in range(num_blocks_to_update):
            seq_pos = offset + i * block_size
            block_idx_in_seq = seq_pos // block_size
            
            # This check is crucial to avoid IndexError when flushing padded batches
            if block_idx_in_seq >= block_tables_np.shape[1]:
                break
                
            physical_block = int(block_tables_np[b, block_idx_in_seq])
            
            start_s = i * block_size
            end_s = min((i + 1) * block_size, num_toks_to_write)
            num_toks = end_s - start_s
            
            if num_toks <= 0:
                continue

            # In MLX, we update blocks one by one using slice_update
            k_pool_out = mx.slice_update(
                k_pool_out, k_new[b, :, start_s:end_s, :][None],
                mx.array([physical_block, 0, 0, 0]), (0, 1, 2, 3)
            )
            v_pool_out = mx.slice_update(
                v_pool_out, v_new[b, :, start_s:end_s, :][None],
                mx.array([physical_block, 0, 0, 0]), (0, 1, 2, 3)
            )

    return k_pool_out, v_pool_out


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
