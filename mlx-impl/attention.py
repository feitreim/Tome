from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
from rmsnorm import RMSNorm
from rope import rotate_half

if TYPE_CHECKING:
    from kvcache import KVCache


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.n_rep = num_heads // num_kv_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array,
        layer_idx: int,
        cache: KVCache | None = None,
    ) -> tuple[mx.array, KVCache | None]:
        b, s, _d = x.shape

        q = self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, s, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, s, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # QK normalization (per-head)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        cos_expanded = cos[None, None, :, :]
        sin_expanded = sin[None, None, :, :]
        q = q * cos_expanded + rotate_half(q) * sin_expanded
        k = k * cos_expanded + rotate_half(k) * sin_expanded

        if cache is not None:
            k, v = cache.update(k, v, layer_idx, self.n_rep)
        else:
            if self.n_rep > 1:
                k = mx.repeat(k, self.n_rep, axis=1)
                v = mx.repeat(v, self.n_rep, axis=1)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.where(mask, attn, mx.finfo(mx.float32).min)
        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(mx.bfloat16)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(b, s, -1)
        return self.o_proj(out), cache
