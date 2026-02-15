import mlx.core as mx


def rope_freqs(head_dim: int, seq_len: int, theta: float = 1000000.0) -> tuple[mx.array, mx.array]:
    freqs = 1.0 / (theta ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
    t = mx.arange(seq_len).astype(mx.float32)
    angles = mx.outer(t, freqs)  # (S, head_dim/2)
    cos = mx.concatenate([mx.cos(angles)] * 2, axis=-1)  # (S, head_dim)
    sin = mx.concatenate([mx.sin(angles)] * 2, axis=-1)  # (S, head_dim)
    return cos, sin


def rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rope(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> tuple[mx.array, mx.array]:
    cos = cos[None, None, :, :]  # (1, 1, S, head_dim)
    sin = sin[None, None, :, :]  # (1, 1, S, head_dim)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k
