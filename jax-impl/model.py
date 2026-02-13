from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from kvcache import KVCache

if TYPE_CHECKING:
    from jaxtyping import Array, BFloat16


def rope_freqs(head_dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2) / head_dim))
    t = jnp.arange(seq_len)
    angles = jnp.outer(t, freqs)  # (S, head_dim/2)
    cos = jnp.concatenate([jnp.cos(angles)] * 2, axis=-1)  # (S, head_dim)
    sin = jnp.concatenate([jnp.sin(angles)] * 2, axis=-1)  # (S, head_dim)
    return cos, sin


def rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(q, k, cos, sin):
    # q, k: (H, S, D)  cos, sin: (S, D)
    cos = cos[None, :, :]  # (1, S, D)
    sin = sin[None, :, :]
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones(dim, dtype=jnp.bfloat16))

    def __call__(self, x: BFloat16[Array, "S D"]) -> BFloat16[Array, "S D"]:
        x_f32 = x.astype(jnp.float32)
        normed = x_f32 * jax.lax.rsqrt(jnp.mean(x_f32**2, axis=-1, keepdims=True) + self.eps)
        return (normed * self.weight.value).astype(jnp.bfloat16)


class MoEMLP(nnx.Module):
    def __init__(self, num_experts, active_experts, dim, intermediate_size, rngs):
        self.topk = active_experts
        self.gate = nnx.Linear(dim, num_experts, rngs=rngs, use_bias=False)
        self.up_proj = nnx.Param(jnp.zeros((num_experts, intermediate_size, dim), dtype="bfloat16"))
        self.down_proj = nnx.Param(jnp.zeros((num_experts, dim, intermediate_size), dtype="bfloat16"))
        self.gate_proj = nnx.Param(jnp.zeros((num_experts, intermediate_size, dim), dtype="bfloat16"))

    def __call__(self, x: BFloat16[Array, "S E"]) -> BFloat16[Array, "S E"]:
        S, D = x.shape

        logits = self.gate(x)  # (S, num_experts)
        g = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(logits.dtype)
        prob, choices = jax.lax.top_k(g, self.topk)  # (S, topk) each

        up_w = self.up_proj.value[choices]  # (S, topk, H, D)
        gate_w = self.gate_proj.value[choices]  # (S, topk, H, D)
        down_w = self.down_proj.value[choices]  # (S, topk, D, H)

        up = jnp.einsum("sd,skhd->skh", x, up_w)  # (S, topk, H)
        gate = jnp.einsum("sd,skhd->skh", x, gate_w)  # (S, topk, H)
        hidden = jax.nn.silu(gate) * up  # (S, topk, H)
        down = jnp.einsum("skh,skdh->skd", hidden, down_w)  # (S, topk, D)

        return (down * prob[..., None]).sum(axis=1)  # (S, D)


class Attention(nnx.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rngs):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.n_rep = num_heads // num_kv_heads
        self.q_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(dim, num_kv_heads * self.head_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(dim, num_kv_heads * self.head_dim, use_bias=False, rngs=rngs)
        self.o_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.q_norm = RMSNorm(dim)
        self.k_norm = RMSNorm(num_kv_heads * self.head_dim)

    def __call__(
        self,
        x: BFloat16[Array, "S D"],
        cos,
        sin,
        mask,
        layer_idx: int,
        cache: KVCache | None = None,
        cur_pos: int = 0,
    ) -> tuple[BFloat16[Array, "S D"], KVCache | None]:
        S, D = x.shape
        q = self.q_norm(self.q_proj(x))  # (S, D)
        k = self.k_norm(self.k_proj(x))  # (S, kv_heads * head_dim)
        v = self.v_proj(x)  # (S, kv_heads * head_dim)

        q = q.reshape(S, self.num_heads, self.head_dim).transpose(1, 0, 2)  # (H, S, D)
        k = k.reshape(S, self.num_kv_heads, self.head_dim).transpose(1, 0, 2)  # (kv_H, S, D)
        v = v.reshape(S, self.num_kv_heads, self.head_dim).transpose(1, 0, 2)  # (kv_H, S, D)

        q, k = apply_rope(q, k, cos, sin)

        # Handle KV cache
        new_cache = cache
        if cache is not None:
            k, v, new_cache = cache.update(k, v, layer_idx, cur_pos, self.n_rep)
        else:
            # Repeat KV heads if using GQA
            if self.n_rep > 1:
                k = jnp.repeat(k, self.n_rep, axis=0)  # (H, S, D)
                v = jnp.repeat(v, self.n_rep, axis=0)  # (H, S, D)

        scale = self.head_dim**-0.5
        attn = (q @ k.transpose(0, 2, 1)) * scale  # (H, S, S_kv)
        attn = jnp.where(mask, attn, jnp.finfo(jnp.float32).min)
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(jnp.bfloat16)

        out = (attn @ v).transpose(1, 0, 2).reshape(S, D)  # (S, D)
        return self.o_proj(out), new_cache


class DecoderLayer(nnx.Module):
    def __init__(self, dim, num_heads, num_kv_heads, num_experts, active_experts, intermediate_size, rngs):
        self.input_norm = RMSNorm(dim)
        self.attn = Attention(dim, num_heads, num_kv_heads, rngs)
        self.post_attn_norm = RMSNorm(dim)
        self.moe = MoEMLP(num_experts, active_experts, dim, intermediate_size, rngs)

    def __call__(self, x, cos, sin, mask, layer_idx: int, cache: KVCache | None = None, cur_pos: int = 0):
        attn_out, new_cache = self.attn(self.input_norm(x), cos, sin, mask, layer_idx, cache, cur_pos)
        x = x + attn_out
        x = x + self.moe(self.post_attn_norm(x))
        return x, new_cache


class OLMoE(nnx.Module):
    def __init__(
        self,
        vocab_size,
        dim,
        num_layers,
        num_heads,
        num_kv_heads,
        num_experts,
        active_experts,
        intermediate_size,
        max_seq_len,
        rngs,
    ):
        self.dim = dim
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.embed = nnx.Embed(vocab_size, dim, rngs=rngs)
        self.layers = nnx.List(
            [
                DecoderLayer(dim, num_heads, num_kv_heads, num_experts, active_experts, intermediate_size, rngs)
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(dim)
        self.lm_head = nnx.Linear(dim, vocab_size, use_bias=False, rngs=rngs)

    def __call__(
        self,
        tokens: BFloat16[Array, "S"],
        cache: KVCache | None = None,
        cur_pos: int = 0,
    ) -> tuple[BFloat16[Array, "S V"], KVCache | None]:
        (S,) = tokens.shape
        x = self.embed(tokens)
        head_dim = x.shape[-1] // self.layers[0].attn.num_heads

        # For cached inference, only compute RoPE for new positions
        if cache is not None:
            cos, sin = rope_freqs(head_dim, self.max_seq_len)
            cos = cos[cur_pos : cur_pos + S]
            sin = sin[cur_pos : cur_pos + S]
            # Mask allows attending to all cached positions + current
            total_len = cur_pos + S
            mask = jnp.tril(jnp.ones((S, total_len), dtype=bool))[None, :, :]
        else:
            cos, sin = rope_freqs(head_dim, S)
            mask = jnp.tril(jnp.ones((S, S), dtype=bool))[None, :, :]

        new_cache = cache
        for i, layer in enumerate(self.layers):
            x, new_cache = layer(x, cos, sin, mask, i, new_cache, cur_pos)

        x = self.norm(x)
        return self.lm_head(x), new_cache
