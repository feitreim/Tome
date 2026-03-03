import mlx.core as mx
import numpy as np
from typing import Optional, Union, List, Dict, Tuple
import time

class PrefixNode:
    def __init__(self, blocks: List[int]):
        self.blocks = blocks  # Physical block indices for the prefix up to this node
        self.children: Dict[int, 'PrefixNode'] = {}
        self.last_access = time.time()

    def update_access(self):
        self.last_access = time.time()

class PrefixCache:
    """Radix tree for prefix caching. Keyed by token sequences."""
    def __init__(self, allocator: 'BlockAllocator'):
        self.allocator = allocator
        self.root = PrefixNode([])
        self.num_cached_tokens = 0

    def lookup(self, tokens: List[int]) -> Tuple[int, List[int]]:
        """
        Find the longest matching prefix for the given tokens.
        Returns (matched_token_count, physical_blocks)
        """
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
        """
        Insert a prefix into the cache.
        """
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

        # Pool: [layer][block] -> (num_kv_heads, block_size, head_dim)
        self.k_pool = [[mx.zeros((num_kv_heads, block_size, head_dim), dtype=dtype) for _ in range(num_blocks)] for _ in range(num_layers)]
        self.v_pool = [[mx.zeros((num_kv_heads, block_size, head_dim), dtype=dtype) for _ in range(num_blocks)] for _ in range(num_layers)]

        self.free_blocks = list(range(num_blocks))
        self.ref_counts = [0] * num_blocks

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
        
        new_block = self.allocate()
        for l in range(self.num_layers):
            self.k_pool[l][new_block] = self.k_pool[l][block_idx]
            self.v_pool[l][new_block] = self.v_pool[l][block_idx]
        
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
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        
        if allocator is None:
            self.allocator = BlockAllocator(
                num_blocks=2048,
                block_size=128,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim
            )
        else:
            self.allocator = allocator
            
        self.block_tables = [[] for _ in range(batch_size)]
        self.offsets = mx.zeros((batch_size,), dtype=mx.int32)

    def __del__(self):
        for b_table in self.block_tables:
            for b in b_table:
                self.allocator.release(b)

    def update(self, k: mx.array, v: mx.array, layer_idx: int) -> tuple[mx.array, mx.array]:
        B, H, S_new, D = k.shape
        block_size = self.allocator.block_size
        
        # We need to handle per-sequence offsets
        # For simplicity, we assume S_new is same for all sequences in batch
        # (This is true for prefill and decode)
        
        current_offsets = np.array(self.offsets)
        needed_lens = current_offsets + S_new
        
        for b_idx in range(B):
            needed_blocks = (needed_lens[b_idx] + block_size - 1) // block_size
            while len(self.block_tables[b_idx]) < needed_blocks:
                self.block_tables[b_idx].append(self.allocator.allocate())
            
            # CoW
            start_b_idx = current_offsets[b_idx] // block_size
            end_b_idx = (needed_lens[b_idx] - 1) // block_size
            for b_table_idx in range(start_b_idx, end_b_idx + 1):
                self.block_tables[b_idx][b_table_idx] = self.allocator.cow_block(self.block_tables[b_idx][b_table_idx])

            # Update blocks
            for s in range(S_new):
                curr_pos = current_offsets[b_idx] + s
                b_table_idx = curr_pos // block_size
                off_in_block = curr_pos % block_size
                physical_block = self.block_tables[b_idx][b_table_idx]
                
                bk = self.allocator.k_pool[layer_idx][physical_block]
                bv = self.allocator.v_pool[layer_idx][physical_block]
                
                self.allocator.k_pool[layer_idx][physical_block] = mx.slice_update(
                    bk,
                    k[b_idx, :, s : s + 1, :],
                    mx.array([0, int(off_in_block), 0]),
                    (0, 1, 2)
                )
                self.allocator.v_pool[layer_idx][physical_block] = mx.slice_update(
                    bv,
                    v[b_idx, :, s : s + 1, :],
                    mx.array([0, int(off_in_block), 0]),
                    (0, 1, 2)
                )

        return self.gather_kv(layer_idx, new_offsets=mx.array(needed_lens))

    def advance(self, num_tokens: Union[int, mx.array]) -> None:
        self.offsets += num_tokens

    def get_seq_len(self, layer_idx: int) -> mx.array:
        # If all offsets are same, return a scalar for efficiency
        if self.batch_size == 1:
            return self.offsets[0]
        # Check if all same
        off_np = np.array(self.offsets)
        if np.all(off_np == off_np[0]):
            return self.offsets[0]
        return self.offsets

    def gather_kv(self, layer_idx: int, new_offsets: Optional[mx.array] = None) -> tuple[mx.array, mx.array]:
        all_k = []
        all_v = []
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
            
        # Pad to max length in batch if they differ
        max_len = int(np.max(offsets))
        padded_k = []
        padded_v = []
        for b_idx in range(self.batch_size):
            curr_k = all_k[b_idx]
            curr_v = all_v[b_idx]
            curr_len = curr_k.shape[1]
            if curr_len < max_len:
                pad_len = max_len - curr_len
                curr_k = mx.concatenate([curr_k, mx.zeros((self.num_kv_heads, pad_len, self.head_dim), dtype=self.allocator.dtype)], axis=1)
                curr_v = mx.concatenate([curr_v, mx.zeros((self.num_kv_heads, pad_len, self.head_dim), dtype=self.allocator.dtype)], axis=1)
            padded_k.append(curr_k)
            padded_v.append(curr_v)
            
        return mx.stack(padded_k), mx.stack(padded_v)
