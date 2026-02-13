from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jaxtyping import Array, BFloat16


class KVCache(NamedTuple):
    k: BFloat16[Array, "_layers _batch _kv_heads _max_seq_len _head_dim"]
    v: BFloat16[Array, "_layers _batch _kv_heads _max_seq_len _head_dim"]

    @classmethod
    def new(
        cls,
        layers: int,
        batch: int,
        max_seq_len: int,
        kv_heads: int,
        head_dim: int,
    ) -> KVCache:
        return cls(
            k=jnp.zeros((layers, batch, kv_heads, max_seq_len, head_dim), dtype=jnp.bfloat16),
            v=jnp.zeros((layers, batch, kv_heads, max_seq_len, head_dim), dtype=jnp.bfloat16),
        )

    def update(
        self,
        k: BFloat16[Array, "_batch _kv_heads _seq_len _head_dim"],
        v: BFloat16[Array, "_batch _kv_heads _seq_len _head_dim"],
        layer_num: int,
        cur_pos: int,
        n_reps: int,
    ) -> tuple[BFloat16[Array, "B KVH SEQ HD"], BFloat16[Array, "B KVH SEQ HD"], KVCache]:
        # Ensure dtype compatibility with cache
        k = k.astype(jnp.bfloat16)
        v = v.astype(jnp.bfloat16)

        ck = jax.lax.dynamic_update_slice(self.k, k[None, ...], (layer_num, 0, 0, cur_pos, 0))
        cv = jax.lax.dynamic_update_slice(self.v, v[None, ...], (layer_num, 0, 0, cur_pos, 0))

        # Retrieve all keys/values up to current position
        # k, v input: (B, kv_heads, S, head_dim)
        # cache: (layers, batch, kv_heads, max_seq_len, head_dim)
        if cur_pos == 0:
            # First token(s), just use the new k, v and repeat for GQA
            keys = jnp.repeat(k, n_reps, axis=1)  # (B, H, S, D)
            values = jnp.repeat(v, n_reps, axis=1)  # (B, H, S, D)
        else:
            # Retrieve cached values: (B, kv_heads, cur_pos+S, head_dim)
            cached_k = ck[layer_num, :, :, : cur_pos + k.shape[2], :]
            cached_v = cv[layer_num, :, :, : cur_pos + v.shape[2], :]
            # Repeat for GQA: (B, kv_heads, seq, head_dim) -> (B, H, seq, head_dim)
            keys = jnp.repeat(cached_k, n_reps, axis=1)
            values = jnp.repeat(cached_v, n_reps, axis=1)

        return keys, values, KVCache(k=ck, v=cv)
