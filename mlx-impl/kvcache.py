import mlx.core as mx
import numpy as np
from typing import Optional, Union, List, Dict, Tuple
import time

from paged_kv import update_paged_kv, gather_paged_kv

class PrefixNode:
    def __init__(self, blocks: List[int]):
        self.blocks = blocks
        self.children: Dict[int, 'PrefixNode'] = {}
        self.last_access = time.time()

    def update_access(self):
        self.last_access = time.time()

class PrefixCache:
    def __init__(self, allocator: 'BlockAllocator'):
        self.allocator = allocator
        self.root = PrefixNode([])
        self.num_cached_tokens = 0

    def lookup(self, tokens: List[int]) -> Tuple[int, List[int]]:
        node = self.root
        matched_tokens = 0
        best_blocks = []
        for i, token in enumerate(tokens):
            if token in node.children:
                node = node.children[token]
                node.update_access()
                matched_tokens = i + 1
                best_blocks = list(node.blocks)
            else:
                break
        for b in best_blocks:
            self.allocator.retain(b)
        return matched_tokens, best_blocks

    def insert(self, tokens: List[int], blocks: List[int]):
        node = self.root
        block_size = self.allocator.block_size
        for i, token in enumerate(tokens):
            if token not in node.children:
                needed_blocks = (i + 1 + block_size - 1) // block_size
                prefix_blocks = blocks[:needed_blocks]
                for b in prefix_blocks:
                    self.allocator.retain(b)
                node.children[token] = PrefixNode(prefix_blocks)
            node = node.children[token]
            node.update_access()
        self.num_cached_tokens = max(self.num_cached_tokens, len(tokens))

class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int, num_layers: int, num_kv_heads: int, head_dim: int, dtype: mx.Dtype = mx.bfloat16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.k_pool = [mx.zeros((num_blocks, num_kv_heads, block_size, head_dim), dtype=dtype) for _ in range(num_layers)]
        self.v_pool = [mx.zeros((num_blocks, num_kv_heads, block_size, head_dim), dtype=dtype) for _ in range(num_layers)]
        self.free_blocks = list(range(num_blocks))
        self.ref_counts = [0] * num_blocks

    def allocate(self) -> int:
        if not self.free_blocks: raise MemoryError("Out of KV blocks")
        b = self.free_blocks.pop()
        self.ref_counts[b] = 1
        return b

    def retain(self, block_idx: int):
        self.ref_counts[block_idx] += 1

    def release(self, block_idx: int):
        if self.ref_counts[block_idx] <= 0: return
        self.ref_counts[block_idx] -= 1
        if self.ref_counts[block_idx] == 0: self.free_blocks.append(block_idx)

    def cow_block(self, block_idx: int) -> int:
        if self.ref_counts[block_idx] <= 1: return block_idx
        new_block = self.allocate()
        for l in range(self.num_layers):
            old_k = self.k_pool[l][block_idx : block_idx + 1]
            old_v = self.v_pool[l][block_idx : block_idx + 1]
            self.k_pool[l] = mx.slice_update(self.k_pool[l], old_k, mx.array([new_block, 0, 0, 0]), (0, 1, 2, 3))
            self.v_pool[l] = mx.slice_update(self.v_pool[l], old_v, mx.array([new_block, 0, 0, 0]), (0, 1, 2, 3))
        self.release(block_idx)
        return new_block

class KVCache:
    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int, max_seq_len: int, batch_size: int = 1, allocator: Optional[BlockAllocator] = None, max_kv_tokens: Optional[int] = None):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        if allocator is None:
            kv_cap = max_kv_tokens if max_kv_tokens is not None else min(max_seq_len, 4096) * batch_size
            block_size = 128
            num_blocks = (kv_cap + block_size - 1) // block_size + batch_size
            self.allocator = BlockAllocator(num_blocks=num_blocks, block_size=block_size, num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim)
        else:
            self.allocator = allocator
        self.block_tables = [[] for _ in range(batch_size)]
        self.offsets = mx.zeros((batch_size,), dtype=mx.int32)
        self._contig_k = [None] * num_layers
        self._contig_v = [None] * num_layers
        self._contig_len = [0] * num_layers
        self._contig_capacity = [0] * num_layers
        self._decode_headroom = 1024
        self._step_block_tables_mx = None
        self._step_S_new = 0

    def __del__(self):
        for b_table in self.block_tables:
            for b in b_table: self.allocator.release(b)

    def advance(self, num_tokens: Union[int, mx.array]) -> None:
        self.offsets += num_tokens

    def get_seq_len(self, layer_idx: int) -> mx.array:
        return self.offsets

    def update(self, k: mx.array, v: mx.array, layer_idx: int) -> tuple[mx.array, mx.array]:
        B, H, S_new, D = k.shape
        block_size = self.allocator.block_size
        if layer_idx == 0 or self._step_S_new != S_new or self._step_block_tables_mx is None:
            self._step_S_new = S_new
            self._step_current_offsets_np = np.array(self.offsets)
            self._step_needed_lens_np = self._step_current_offsets_np + S_new
            self._step_max_len = int(np.max(self._step_needed_lens_np))
            for b_idx in range(B):
                needed_blocks = (self._step_needed_lens_np[b_idx] + block_size - 1) // block_size
                while len(self.block_tables[b_idx]) < needed_blocks:
                    self.block_tables[b_idx].append(self.allocator.allocate())
                start_b_idx = self._step_current_offsets_np[b_idx] // block_size
                end_b_idx = (self._step_needed_lens_np[b_idx] - 1) // block_size
                for b_table_idx in range(start_b_idx, end_b_idx + 1):
                    self.block_tables[b_idx][b_table_idx] = self.allocator.cow_block(self.block_tables[b_idx][b_table_idx])
            max_blocks_in_batch = max(len(bt) for bt in self.block_tables)
            flat_block_tables = np.zeros((B, max_blocks_in_batch), dtype=np.int32)
            for b_idx, bt in enumerate(self.block_tables):
                flat_block_tables[b_idx, : len(bt)] = bt
            self._step_block_tables_mx = mx.array(flat_block_tables)
            self._step_max_blocks = max_blocks_in_batch

        k_dummy, v_dummy = update_paged_kv(k, v, self._step_block_tables_mx, self.offsets, self.allocator.k_pool[layer_idx], self.allocator.v_pool[layer_idx], block_size)
        k_pool_dep = self.allocator.k_pool[layer_idx] + k_dummy.astype(k.dtype).reshape(1, 1, 1, 1) * 0
        v_pool_dep = self.allocator.v_pool[layer_idx] + v_dummy.astype(v.dtype).reshape(1, 1, 1, 1) * 0

        cur_len = self._contig_len[layer_idx]
        new_len = cur_len + S_new
        cap = self._contig_capacity[layer_idx]

        if cur_len == 0 or S_new > 1 or new_len > cap:
            kg, vg = gather_paged_kv(self._step_block_tables_mx, self.offsets + S_new, k_pool_dep, v_pool_dep, self._step_max_len, self._step_max_blocks, block_size, B, H, D)
            self._contig_k[layer_idx] = kg
            self._contig_v[layer_idx] = vg
            self._contig_len[layer_idx] = self._step_max_len
            self._contig_capacity[layer_idx] = self._step_max_len
        else:
            # Incremental: just concatenate. Very fast in MLX.
            self._contig_k[layer_idx] = mx.concatenate([self._contig_k[layer_idx], k], axis=2)
            self._contig_v[layer_idx] = mx.concatenate([self._contig_v[layer_idx], v], axis=2)
            # Maintain pool update dependency
            self._contig_k[layer_idx] = self._contig_k[layer_idx] + k_dummy.astype(k.dtype).reshape(1, 1, 1, 1) * 0
            self._contig_v[layer_idx] = self._contig_v[layer_idx] + v_dummy.astype(v.dtype).reshape(1, 1, 1, 1) * 0
            self._contig_len[layer_idx] = new_len


        seq_len = self._contig_len[layer_idx]
        return self._contig_k[layer_idx][:, :, :seq_len, :], self._contig_v[layer_idx][:, :, :seq_len, :]

    def gather_kv(self, layer_idx: int, new_offsets: Optional[mx.array] = None) -> tuple[mx.array, mx.array]:
        all_k, all_v = [], []
        offsets = np.array(new_offsets) if new_offsets is not None else np.array(self.offsets)
        for b_idx in range(self.batch_size):
            indices = self.block_tables[b_idx]
            if not indices:
                all_k.append(mx.zeros((self.num_kv_heads, 0, self.head_dim), dtype=self.allocator.dtype))
                all_v.append(mx.zeros((self.num_kv_heads, 0, self.head_dim), dtype=self.allocator.dtype))
                continue
            k_blocks = [self.allocator.k_pool[layer_idx][i] for i in indices]
            v_blocks = [self.allocator.v_pool[layer_idx][i] for i in indices]
            k_seq = mx.concatenate(k_blocks, axis=1)
            v_seq = mx.concatenate(v_blocks, axis=1)
            all_k.append(k_seq[:, :int(offsets[b_idx]), :])
            all_v.append(v_seq[:, :int(offsets[b_idx]), :])
        max_len = int(np.max(offsets))
        padded_k, padded_v = [], []
        for b_idx in range(self.batch_size):
            curr_k, curr_v = all_k[b_idx], all_v[b_idx]
            curr_len = curr_k.shape[1]
            if curr_len < max_len:
                pad_len = max_len - curr_len
                curr_k = mx.concatenate([curr_k, mx.zeros((self.num_kv_heads, pad_len, self.head_dim), dtype=self.allocator.dtype)], axis=1)
                curr_v = mx.concatenate([curr_v, mx.zeros((self.num_kv_heads, pad_len, self.head_dim), dtype=self.allocator.dtype)], axis=1)
            padded_k.append(curr_k); padded_v.append(curr_v)
        return mx.stack(padded_k), mx.stack(padded_v)
