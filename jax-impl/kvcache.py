from typing import TYPE_CHECKING, NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jaxtyping import Array, BFloat16


class KVCache(NamedTuple):
    k: BFloat16[Array, "L S H D"]
    v: BFloat16[Array, "L S H D"]

    @classmethod
    def new(
        cls,
        layers: int,
        max_seq_len: int,
        kv_heads: int,
        head_dim: int,
    ) -> KVCache:
        return cls(
            k=jnp.zeros((layers, max_seq_len, kv_heads, head_dim), dtype=jnp.bfloat16),
            v=jnp.zeros((layers, max_seq_len, kv_heads, head_dim), dtype=jnp.bfloat16),
        )

    def update(
        self,
        k: BFloat16[Array, "H S D"],
        v: BFloat16[Array, "H S D"],
        layer_num: int,
        cur_pos: int,
        n_reps: int,
    ) -> tuple[BFloat16[Array, "H S D"], BFloat16[Array, "H S D"], KVCache]:
        # k, v input: (kv_heads, S, head_dim)
        # cache: (layers, max_seq_len, kv_heads, head_dim)
        # Need to transpose and insert
        k_transposed = k.transpose(1, 0, 2).astype(jnp.bfloat16)  # (S, kv_heads, head_dim)
        v_transposed = v.transpose(1, 0, 2).astype(jnp.bfloat16)  # (S, kv_heads, head_dim)

        ck = jax.lax.dynamic_update_slice(self.k, k_transposed[None, :, :, :], (layer_num, cur_pos, 0, 0))
        cv = jax.lax.dynamic_update_slice(self.v, v_transposed[None, :, :, :], (layer_num, cur_pos, 0, 0))

        # Retrieve all keys/values up to current position
        if cur_pos == 0:
            # First token, just use the new k, v
            keys = jnp.repeat(k, n_reps, axis=0)  # (H, S, D)
            values = jnp.repeat(v, n_reps, axis=0)  # (H, S, D)
        else:
            # Retrieve cached values: (S_total, kv_heads, head_dim)
            cached_k = ck[layer_num, : cur_pos + k.shape[1], :, :]
            cached_v = cv[layer_num, : cur_pos + v.shape[1], :, :]
            # Transpose back and repeat: (kv_heads, S_total, head_dim) -> (H, S_total, D)
            keys = jnp.repeat(cached_k.transpose(1, 0, 2), n_reps, axis=0)
            values = jnp.repeat(cached_v.transpose(1, 0, 2), n_reps, axis=0)

        return keys, values, KVCache(k=ck, v=cv)
