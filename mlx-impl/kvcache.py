import mlx.core as mx


class KVCache:
    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int, max_seq_len: int):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.keys: list[mx.array | None] = [None] * num_layers
        self.values: list[mx.array | None] = [None] * num_layers
        self.offset = 0

    def update(
        self,
        k: mx.array,
        v: mx.array,
        layer_idx: int,
        n_rep: int,
    ) -> tuple[mx.array, mx.array]:
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = k
            self.values[layer_idx] = v
        else:
            self.keys[layer_idx] = mx.concatenate([self.keys[layer_idx], k], axis=2)
            self.values[layer_idx] = mx.concatenate([self.values[layer_idx], v], axis=2)

        full_k = self.keys[layer_idx]
        full_v = self.values[layer_idx]

        if n_rep > 1:
            full_k = mx.repeat(full_k, n_rep, axis=1)
            full_v = mx.repeat(full_v, n_rep, axis=1)

        return full_k, full_v

    def advance(self, num_tokens: int):
        self.offset += num_tokens
