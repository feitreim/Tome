from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
from attention import Attention
from mlp import SwiGLU
from rmsnorm import RMSNorm

if TYPE_CHECKING:
    from kvcache import KVCache


class DecoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, head_dim: int, intermediate_size: int, eps: float):
        super().__init__()
        self.input_layernorm = RMSNorm(dim, eps=eps)
        self.self_attn = Attention(dim, num_heads, num_kv_heads, head_dim)
        self.post_attention_layernorm = RMSNorm(dim, eps=eps)
        self.mlp = SwiGLU(dim, intermediate_size)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array,
        layer_idx: int,
        cache: KVCache | None = None,
    ) -> tuple[mx.array, KVCache | None]:
        attn_out, cache = self.self_attn(self.input_layernorm(x), cos, sin, mask, layer_idx, cache)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, cache
