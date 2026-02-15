from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
from decoder import DecoderLayer
from rmsnorm import RMSNorm
from rope import rope_freqs

if TYPE_CHECKING:
    from kvcache import KVCache


class Qwen3(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        max_seq_len: int,
        rope_theta: float = 1000000.0,
        eps: float = 1e-6,
        tie_word_embeddings: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings

        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = [
            DecoderLayer(dim, num_heads, num_kv_heads, head_dim, intermediate_size, eps) for _ in range(num_layers)
        ]
        self.norm = RMSNorm(dim, eps=eps)
        if not tie_word_embeddings:
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def __call__(
        self,
        tokens: mx.array,
        cache: KVCache | None = None,
        cur_pos: int = 0,
    ) -> tuple[mx.array, KVCache | None]:
        _b, s = tokens.shape
        x = self.embed_tokens(tokens)

        if cache is not None:
            cos, sin = rope_freqs(self.head_dim, self.max_seq_len, self.rope_theta)
            cos = cos[cur_pos : cur_pos + s]
            sin = sin[cur_pos : cur_pos + s]
            total_len = cur_pos + s
            mask = mx.expand_dims(mx.expand_dims(mx.tril(mx.ones((s, total_len))), axis=0), axis=0).astype(mx.bool_)
        else:
            cos, sin = rope_freqs(self.head_dim, s, self.rope_theta)
            mask = mx.expand_dims(mx.expand_dims(mx.tril(mx.ones((s, s))), axis=0), axis=0).astype(mx.bool_)

        for i, layer in enumerate(self.layers):
            x, cache = layer(x, cos, sin, mask, i, cache)

        x = self.norm(x)

        logits = x @ self.embed_tokens.weight.T if self.tie_word_embeddings else self.lm_head(x)

        return logits, cache
