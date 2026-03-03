import mlx.core as mx
import numpy as np
from typing import Optional, Union, List, Dict, Tuple
import time

from paged_kv import update_paged_kv, gather_paged_kv


class PrefixNode:
    def __init__(self, blocks: List[int]):
        self.blocks = blocks
        self.children: Dict[int, "PrefixNode"] = {}
        self.last_access = time.time()

    def update_access(self):
        self.last_access = time.time()


class PrefixCache:
    def __init__(self, allocator: "BlockAllocator"):
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
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.bfloat16,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        # Pools are lazily allocated on first flush_to_pool call
        self.k_pool: Optional[List[mx.array]] = None
        self.v_pool: Optional[List[mx.array]] = None
        self.free_blocks = list(range(num_blocks))
        self.ref_counts = [0] * num_blocks

    def _ensure_pools(self):
        if self.k_pool is None:
            self.k_pool = [
                mx.zeros((self.num_blocks, self.num_kv_heads, self.block_size, self.head_dim), dtype=self.dtype)
                for _ in range(self.num_layers)
            ]
            self.v_pool = [
                mx.zeros((self.num_blocks, self.num_kv_heads, self.block_size, self.head_dim), dtype=self.dtype)
                for _ in range(self.num_layers)
            ]

    def allocate(self) -> int:
        if not self.free_blocks:
            raise MemoryError("Out of KV blocks")
        b = self.free_blocks.pop()
        self.ref_counts[b] = 1
        return b

    def retain(self, block_idx: int):
        self.ref_counts[block_idx] += 1

    def release(self, block_idx: int):
        if self.ref_counts[block_idx] <= 0:
            return
        self.ref_counts[block_idx] -= 1
        if self.ref_counts[block_idx] == 0:
            self.free_blocks.append(block_idx)

    def cow_block(self, block_idx: int) -> int:
        if self.ref_counts[block_idx] <= 1:
            return block_idx
        self._ensure_pools()
        new_block = self.allocate()
        for l in range(self.num_layers):
            old_k = self.k_pool[l][block_idx : block_idx + 1]
            old_v = self.v_pool[l][block_idx : block_idx + 1]
            self.k_pool[l] = mx.slice_update(self.k_pool[l], old_k, mx.array([new_block, 0, 0, 0]), (0, 1, 2, 3))
            self.v_pool[l] = mx.slice_update(self.v_pool[l], old_v, mx.array([new_block, 0, 0, 0]), (0, 1, 2, 3))
        self.release(block_idx)
        return new_block


class KVCache:
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        batch_size: int = 1,
        allocator: Optional[BlockAllocator] = None,
        max_kv_tokens: Optional[int] = None,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        if allocator is None:
            kv_cap = max_kv_tokens if max_kv_tokens is not None else min(max_seq_len, 4096) * batch_size
            block_size = 128
            num_blocks = (kv_cap + block_size - 1) // block_size + batch_size
            self.allocator = BlockAllocator(
                num_blocks=num_blocks,
                block_size=block_size,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
        else:
            self.allocator = allocator
        self.block_tables: List[List[int]] = [[] for _ in range(batch_size)]
        self.offsets = mx.zeros((batch_size,), dtype=mx.int32)
        # Contiguous cache for fast attention (source of truth during generation)
        self._keys: List[Optional[mx.array]] = [None] * num_layers
        self._values: List[Optional[mx.array]] = [None] * num_layers

    def __del__(self):
        for b_table in self.block_tables:
            for b in b_table:
                self.allocator.release(b)

    def advance(self, num_tokens: Union[int, mx.array]) -> None:
        self.offsets += num_tokens

    def get_seq_len(self, layer_idx: int) -> mx.array:
        return self.offsets

    def update(self, k: mx.array, v: mx.array, layer_idx: int) -> tuple[mx.array, mx.array]:
        if self._keys[layer_idx] is None:
            self._keys[layer_idx] = k
            self._values[layer_idx] = v
        else:
            self._keys[layer_idx] = mx.concatenate([self._keys[layer_idx], k], axis=2)
            self._values[layer_idx] = mx.concatenate([self._values[layer_idx], v], axis=2)
        return self._keys[layer_idx], self._values[layer_idx]

    def flush_to_pool(self):
        """Write contiguous cache data to paged pool for prefix caching.
        Call this after generation when you want to persist KV data for future prefix sharing.
        Requires mx.eval() before calling to ensure cache data is materialized."""
        self.allocator._ensure_pools()
        block_size = self.allocator.block_size
        offsets_np = np.array(self.offsets)
        B = self.batch_size
        for b_idx in range(B):
            seq_len = int(offsets_np[b_idx])
            needed_blocks = (seq_len + block_size - 1) // block_size
            while len(self.block_tables[b_idx]) < needed_blocks:
                self.block_tables[b_idx].append(self.allocator.allocate())
        max_blocks_in_batch = max(len(bt) for bt in self.block_tables) if self.block_tables else 0
        if max_blocks_in_batch == 0:
            return
        flat_bt = np.zeros((B, max_blocks_in_batch), dtype=np.int32)
        for b_idx, bt in enumerate(self.block_tables):
            flat_bt[b_idx, : len(bt)] = bt
        block_tables_mx = mx.array(flat_bt)
        # Zero offsets for the scatter (we're writing from position 0)
        zero_offsets = mx.zeros((B,), dtype=mx.int32)
        for layer_idx in range(self.num_layers):
            if self._keys[layer_idx] is None:
                continue
            update_paged_kv(
                self._keys[layer_idx],
                self._values[layer_idx],
                block_tables_mx,
                zero_offsets,
                self.allocator.k_pool[layer_idx],
                self.allocator.v_pool[layer_idx],
                block_size,
            )

    def gather_kv(self, layer_idx: int, new_offsets: Optional[mx.array] = None) -> tuple[mx.array, mx.array]:
        """Gather KV from paged pool. Used for restoring from prefix cache."""
        self.allocator._ensure_pools()
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
            all_k.append(k_seq[:, : int(offsets[b_idx]), :])
            all_v.append(v_seq[:, : int(offsets[b_idx]), :])
        max_len = int(np.max(offsets))
        padded_k, padded_v = [], []
        for b_idx in range(self.batch_size):
            curr_k, curr_v = all_k[b_idx], all_v[b_idx]
            curr_len = curr_k.shape[1]
            if curr_len < max_len:
                pad_len = max_len - curr_len
                curr_k = mx.concatenate(
                    [curr_k, mx.zeros((self.num_kv_heads, pad_len, self.head_dim), dtype=self.allocator.dtype)], axis=1
                )
                curr_v = mx.concatenate(
                    [curr_v, mx.zeros((self.num_kv_heads, pad_len, self.head_dim), dtype=self.allocator.dtype)], axis=1
                )
            padded_k.append(curr_k)
            padded_v.append(curr_v)
        return mx.stack(padded_k), mx.stack(padded_v)
