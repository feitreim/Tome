import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(dim)

    def __call__(self, x: mx.array) -> mx.array:
        x_f32 = x.astype(mx.float32)
        normed = x_f32 * mx.rsqrt(mx.mean(x_f32 * x_f32, axis=-1, keepdims=True) + self.eps)
        return (normed * self.weight).astype(mx.bfloat16)
