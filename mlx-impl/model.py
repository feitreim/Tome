from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from kvcache import KVCache


# ---------------------------------------------------------------------------
# Metal Kernels
# ---------------------------------------------------------------------------

_fused_log_softmax_source = """
    constexpr int M = 4;
    constexpr int block = 1024 * M;
    constexpr int full_blocks = V / block;
    constexpr int extra = V - full_blocks * block;
    threadgroup float shared[32];
    uint row = threadgroup_position_in_grid.y;
    uint tid = thread_index_in_threadgroup;
    uint simd_lane_id = thread_index_in_simdgroup;
    uint simd_group_id = simdgroup_index_in_threadgroup;
    logits += row * V; out += row * V;
    float inv_temp = 1.0f / temp[0];
    float thread_max = -1e30f;
    int offset = tid * M;
    for (int i = 0; i < full_blocks; i++) {
        for (int j = 0; j < M; j++) thread_max = max(thread_max, static_cast<float>(logits[offset+j]) * inv_temp);
        offset += block;
    }
    if (extra > 0) {
        for (int j = 0; j < M; j++) if (offset+j < V) thread_max = max(thread_max, static_cast<float>(logits[offset+j]) * inv_temp);
    }
    float simd_max_val = simd_max(thread_max);
    if (simd_lane_id == 0) shared[simd_group_id] = simd_max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) { float v = shared[simd_lane_id]; v = simd_max(v); if (simd_lane_id == 0) shared[0] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = shared[0];
    float sum_exp = 0.0f; offset = tid * M;
    for (int i = 0; i < full_blocks; i++) {
        for (int j = 0; j < M; j++) sum_exp += metal::fast::exp(static_cast<float>(logits[offset+j]) * inv_temp - row_max);
        offset += block;
    }
    if (extra > 0) {
        for (int j = 0; j < M; j++) if (offset+j < V) sum_exp += metal::fast::exp(static_cast<float>(logits[offset+j]) * inv_temp - row_max);
    }
    sum_exp = simd_sum(sum_exp);
    if (simd_lane_id == 0) shared[simd_group_id] = sum_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) { float v = shared[simd_lane_id]; v = simd_sum(v); if (simd_lane_id == 0) shared[0] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float log_sum_exp = metal::fast::log(shared[0]);
    T lse = static_cast<T>(row_max + log_sum_exp);
    offset = tid * M;
    for (int i = 0; i < full_blocks; i++) {
        for (int j = 0; j < M; j++) out[offset+j] = static_cast<T>(static_cast<float>(logits[offset+j]) * inv_temp) - lse;
        offset += block;
    }
    if (extra > 0) {
        for (int j = 0; j < M; j++) if (offset+j < V) out[offset+j] = static_cast<T>(static_cast<float>(logits[offset+j]) * inv_temp) - lse;
    }
"""

_fused_log_softmax_kernel = mx.fast.metal_kernel(
    name="fused_log_softmax",
    input_names=["logits", "temp"],
    output_names=["out"],
    source=_fused_log_softmax_source,
    ensure_row_contiguous=True,
)


def fused_log_softmax(logits: mx.array, temperature: float = 1.0) -> mx.array:
    """Fast non-differentiable log-softmax kernel for rollouts."""
    V = logits.shape[-1]
    flat = logits.reshape(-1, V)
    res = _fused_log_softmax_kernel(
        inputs=[flat, mx.array([temperature], dtype=mx.float32)],
        output_shapes=[flat.shape],
        output_dtypes=[logits.dtype],
        template=[("T", logits.dtype), ("V", V)],
        grid=(1024, flat.shape[0], 1),
        threadgroup=(1024, 1, 1),
    )[0]
    return res.reshape(logits.shape)


# ---------------------------------------------------------------------------
# Fused reshape + transpose + RMSNorm + RoPE Metal kernel
# ---------------------------------------------------------------------------
# Fuses four ops into a single GPU pass:
#   1. Reshape (B, S, NH*HD) -> (B, S, NH, HD)
#   2. Transpose -> (B, NH, S, HD)
#   3. Per-head RMSNorm
#   4. Non-traditional (NeoX-style) RoPE
# One threadgroup per (batch, head, seq_pos), HD threads per group.

_norm_rope_kernels: dict[tuple[int, float], Any] = {}


def _get_norm_rope_kernel(head_dim: int, eps: float = 1e-6, rope_base: float = 1000000.0) -> Any:
    cache_key = (head_dim, rope_base)
    if cache_key in _norm_rope_kernels:
        return _norm_rope_kernels[cache_key]

    n_simd = head_dim // 32
    half = head_dim // 2

    source = f"""
        threadgroup float partial_sums[{n_simd}];
        threadgroup float normed_vals[{head_dim}];

        uint tid = thread_position_in_threadgroup.x;
        uint gid = threadgroup_position_in_grid.x;

        uint S = seq_len[0];
        uint seq = gid % S;
        uint head = (gid / S) % NH;
        uint batch = gid / (S * NH);

        // Read from (B, S, NH*HD) contiguous layout
        uint in_idx = batch * (S * NH * {head_dim}) + seq * (NH * {head_dim}) + head * {head_dim} + tid;
        float val = float(inp[in_idx]);

        // RMSNorm: SIMD reduction + cross-group shared mem reduction
        float simd_sq = simd_sum(val * val);
        if (thread_index_in_simdgroup == 0) {{
            partial_sums[tid / threads_per_simdgroup] = simd_sq;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float total_sq = 0.0f;
        for (uint i = 0; i < {n_simd}; i++) {{
            total_sq += partial_sums[i];
        }}
        float normed = val * metal::rsqrt(total_sq / {float(head_dim)}f + {eps}f) * float(norm_w[tid]);

        // Store normalised values for RoPE partner access
        normed_vals[tid] = normed;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // RoPE (non-traditional / NeoX-style: pairs [i, i+d/2])
        uint pos = rope_offset[0] + seq;
        uint rope_dim = (tid < {half}) ? tid : (tid - {half});
        float freq = 1.0f / metal::pow({rope_base}f, float(2 * rope_dim) / {float(head_dim)}f);
        float angle = float(pos) * freq;
        float cos_val = metal::precise::cos(angle);
        float sin_val = metal::precise::sin(angle);

        float result;
        if (tid < {half}) {{
            result = normed_vals[tid] * cos_val - normed_vals[tid + {half}] * sin_val;
        }} else {{
            result = normed_vals[tid] * cos_val + normed_vals[tid - {half}] * sin_val;
        }}

        // Write to (B, NH, S, HD) transposed layout
        uint out_idx = batch * (NH * S * {head_dim}) + head * (S * {head_dim}) + seq * {head_dim} + tid;
        out[out_idx] = static_cast<T>(result);
    """

    kernel = mx.fast.metal_kernel(
        name="fused_norm_rope",
        input_names=["inp", "norm_w", "seq_len", "rope_offset"],
        output_names=["out"],
        source=source,
    )
    _norm_rope_kernels[cache_key] = kernel
    return kernel


def fused_norm_rope(
    proj_out: mx.array,
    norm_weight: mx.array,
    num_heads: int,
    head_dim: int,
    offset: int,
    rope_theta: float,
) -> mx.array:
    b, s, _ = proj_out.shape
    kernel = _get_norm_rope_kernel(head_dim, rope_base=rope_theta)
    return kernel(
        inputs=[proj_out, norm_weight, mx.array([s], dtype=mx.uint32), mx.array([offset], dtype=mx.uint32)],
        template=[("NH", num_heads), ("T", proj_out.dtype)],
        grid=(b * num_heads * s * head_dim, 1, 1),
        threadgroup=(head_dim, 1, 1),
        output_shapes=[(b, num_heads, s, head_dim)],
        output_dtypes=[proj_out.dtype],
        stream=mx.gpu,
    )[0]


# ---------------------------------------------------------------------------
# Fused reshape + transpose + RoPE Metal kernel (no norm)
# ---------------------------------------------------------------------------
# Fuses three ops into a single GPU pass:
#   1. Reshape (B, S, NH*HD) -> (B, S, NH, HD)
#   2. Transpose -> (B, NH, S, HD)
#   3. RoPE (non-interleaved Llama-style: pairs [i, i+d/2])
# One threadgroup per (batch, head, seq_pos), HD threads per group.

_rope_kernels: dict[tuple[int, float], Any] = {}


def _get_rope_kernel(head_dim: int, rope_base: float = 1000000.0) -> Any:
    cache_key = (head_dim, rope_base)
    if cache_key in _rope_kernels:
        return _rope_kernels[cache_key]

    half = head_dim // 2

    source = f"""
        threadgroup float vals[{head_dim}];

        uint tid = thread_position_in_threadgroup.x;
        uint gid = threadgroup_position_in_grid.x;

        uint S = seq_len[0];
        uint seq = gid % S;
        uint head = (gid / S) % NH;
        uint batch = gid / (S * NH);

        // Read from (B, S, NH*HD) contiguous layout
        uint in_idx = batch * (S * NH * {head_dim}) + seq * (NH * {head_dim}) + head * {head_dim} + tid;
        float val = float(inp[in_idx]);

        // Store values for RoPE partner access
        vals[tid] = val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // RoPE (non-interleaved Llama-style: pairs [i, i+d/2])
        uint pos = rope_offset[0] + seq;
        uint rope_dim = (tid < {half}) ? tid : (tid - {half});
        float freq = 1.0f / metal::pow({rope_base}f, float(2 * rope_dim) / {float(head_dim)}f);
        float angle = float(pos) * freq;
        float cos_val = metal::precise::cos(angle);
        float sin_val = metal::precise::sin(angle);

        float result;
        if (tid < {half}) {{
            result = vals[tid] * cos_val - vals[tid + {half}] * sin_val;
        }} else {{
            result = vals[tid] * cos_val + vals[tid - {half}] * sin_val;
        }}

        // Write to (B, NH, S, HD) transposed layout
        uint out_idx = batch * (NH * S * {head_dim}) + head * (S * {head_dim}) + seq * {head_dim} + tid;
        out[out_idx] = static_cast<T>(result);
    """

    kernel = mx.fast.metal_kernel(
        name="fused_rope",
        input_names=["inp", "seq_len", "rope_offset"],
        output_names=["out"],
        source=source,
    )
    _rope_kernels[cache_key] = kernel
    return kernel


def fused_rope(
    proj_out: mx.array,
    num_heads: int,
    head_dim: int,
    offset: int,
    rope_theta: float,
) -> mx.array:
    b, s, _ = proj_out.shape
    kernel = _get_rope_kernel(head_dim, rope_base=rope_theta)
    return kernel(
        inputs=[proj_out, mx.array([s], dtype=mx.uint32), mx.array([offset], dtype=mx.uint32)],
        template=[("NH", num_heads), ("T", proj_out.dtype)],
        grid=(b * num_heads * s * head_dim, 1, 1),
        threadgroup=(head_dim, 1, 1),
        output_shapes=[(b, num_heads, s, head_dim)],
        output_dtypes=[proj_out.dtype],
        stream=mx.gpu,
    )[0]


def _rope_workaround(x, dims, theta, offset, traditional=False):
    # mx.fast.rope is buggy for S=1 with B>1 when offset is a scalar.
    # If offset is an array of shape (B,), it works correctly.
    if isinstance(offset, mx.array) and offset.ndim > 0:
        return mx.fast.rope(x, dims, traditional=traditional, base=theta, scale=1.0, offset=offset)
    
    # Fallback/Workaround for scalar offset
    if x.shape[2] == 1 and x.shape[0] > 1:
        if isinstance(offset, mx.array):
            offsets = mx.full((x.shape[0],), offset, dtype=mx.int32)
        else:
            offsets = mx.array([offset] * x.shape[0], dtype=mx.int32)
        return mx.fast.rope(x, dims, traditional=traditional, base=theta, scale=1.0, offset=offsets)
    
    return mx.fast.rope(x, dims, traditional=traditional, base=theta, scale=1.0, offset=offset)


# ---------------------------------------------------------------------------
# Baseline (separate MLX ops) for benchmarking
# ---------------------------------------------------------------------------


def baseline_norm_rope(
    proj_out: mx.array,
    norm_weight: mx.array,
    eps: float,
    num_heads: int,
    head_dim: int,
    offset: int,
    rope_theta: float,
    rope_traditional: bool,
) -> mx.array:
    b, s, _ = proj_out.shape
    x = proj_out.reshape(b, s, num_heads, head_dim).transpose(0, 2, 1, 3)
    x = mx.fast.rms_norm(x, norm_weight, eps)
    return _rope_workaround(x, head_dim, rope_theta, offset, traditional=rope_traditional)


def baseline_rope(
    proj_out: mx.array,
    num_heads: int,
    head_dim: int,
    offset: int,
    rope_theta: float,
    rope_traditional: bool,
) -> mx.array:
    b, s, _ = proj_out.shape
    x = proj_out.reshape(b, s, num_heads, head_dim).transpose(0, 2, 1, 3)
    return _rope_workaround(x, head_dim, rope_theta, offset, traditional=rope_traditional)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(dim)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class MLP(nn.Module):
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(dim, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gu = self.gate_up_proj(x)
        gate, up = mx.split(gu, 2, axis=-1)
        return self.down_proj(nn.silu(gate) * up)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        use_qk_norm: bool,
        eps: float,
        rope_theta: float,
        rope_traditional: bool,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_qk_norm = use_qk_norm
        self.eps = eps
        self.rope_theta = rope_theta
        self.rope_traditional = rope_traditional
        self.scale = head_dim**-0.5

        self.qkv_proj = nn.Linear(dim, (num_heads + 2 * num_kv_heads) * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)
        if use_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=eps)
            self.k_norm = RMSNorm(head_dim, eps=eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def __call__(
        self,
        x: mx.array,
        mask: str | mx.array,
        layer_idx: int,
        cache: KVCache | None = None,
    ) -> tuple[mx.array, KVCache | None]:
        b, s, _ = x.shape
        offset = 0 if cache is None else cache.get_seq_len(layer_idx)

        # Fused projection for efficiency
        qkv = self.qkv_proj(x)

        # Split into Q, K, V
        q_size = self.num_heads * self.head_dim
        k_size = self.num_kv_heads * self.head_dim
        q = qkv[..., :q_size]
        k = qkv[..., q_size : q_size + k_size]
        v = qkv[..., q_size + k_size :]

        if self.use_qk_norm:
            # Fused kernel combines reshape, transpose, RMSNorm and RoPE into a single pass
            from model import fused_norm_rope
            q = fused_norm_rope(q, self.q_norm.weight, self.num_heads, self.head_dim, offset, self.rope_theta)
            k = fused_norm_rope(k, self.k_norm.weight, self.num_kv_heads, self.head_dim, offset, self.rope_theta)
        else:
            # Unfused fallback if QK norm is disabled
            q = q.reshape(b, s, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(b, s, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            q = _rope_workaround(q, self.head_dim, self.rope_theta, offset, traditional=self.rope_traditional)
            k = _rope_workaround(k, self.head_dim, self.rope_theta, offset, traditional=self.rope_traditional)

        v = v.reshape(b, s, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            k, v = cache.update(k, v, layer_idx)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(b, s, -1)
        return self.o_proj(out), cache


class DecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        eps: float,
        *,
        use_qk_norm: bool,
        rope_theta: float,
        rope_traditional: bool,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(dim, eps=eps)
        self.self_attn = Attention(
            dim,
            num_heads,
            num_kv_heads,
            head_dim,
            use_qk_norm=use_qk_norm,
            eps=eps,
            rope_theta=rope_theta,
            rope_traditional=rope_traditional,
        )
        self.post_attention_layernorm = RMSNorm(dim, eps=eps)
        self.mlp = MLP(dim, intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: str | mx.array,
        layer_idx: int,
        cache: KVCache | None = None,
    ) -> tuple[mx.array, KVCache | None]:
        attn_out, cache = self.self_attn(self.input_layernorm(x), mask, layer_idx, cache)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, cache


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
        use_qk_norm: bool = True,
        rope_traditional: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        self.use_qk_norm = use_qk_norm
        self.rope_traditional = rope_traditional

        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = [
            DecoderLayer(
                dim,
                num_heads,
                num_kv_heads,
                head_dim,
                intermediate_size,
                eps,
                use_qk_norm=use_qk_norm,
                rope_theta=rope_theta,
                rope_traditional=rope_traditional,
            )
            for _ in range(num_layers)
        ]
        self.norm = RMSNorm(dim, eps=eps)
        if not tie_word_embeddings:
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def __call__(
        self,
        tokens: mx.array,
        cache: KVCache | None = None,
    ) -> tuple[mx.array, KVCache | None]:
        _b, s = tokens.shape
        x = self.embed_tokens(tokens)

        # Fast kernels use "causal" mask string for automatic causal masking
        mask = "causal" if s > 1 else None

        for i, layer in enumerate(self.layers):
            x, cache = layer(x, mask, i, cache)

        x = self.norm(x)

        logits = x @ self.embed_tokens.weight.T if self.tie_word_embeddings else self.lm_head(x)

        return logits, cache
