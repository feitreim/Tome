import sys
import time
import os
sys.path.append(os.path.join(os.getcwd(), "mlx-impl"))

import mlx.core as mx
from kvcache import BlockAllocator, KVCache
from model import Qwen3

VOCAB_SIZE = 151936
DIM = 1024
NUM_LAYERS = 28
NUM_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE_SIZE = 3072
MAX_SEQ_LEN = 40960
ROPE_THETA = 1000000.0
EPS = 1e-6


def build_our_model():
    model = Qwen3(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        intermediate_size=INTERMEDIATE_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        rope_theta=ROPE_THETA,
        eps=EPS,
        tie_word_embeddings=True,
        use_qk_norm=True,
        rope_traditional=False,
    )
    return model


def profile_e2e():
    """End-to-end per-step latency (the only honest measurement without GPU capture)."""
    model = build_our_model()
    mx.eval(model.parameters())

    batch_size = 32
    prompt_len = 32
    gen_tokens = 50

    max_seq = prompt_len + gen_tokens + 10
    num_blocks = (max_seq * batch_size + 127) // 128 + batch_size  # headroom
    allocator = BlockAllocator(num_blocks, 128, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM)
    cache = KVCache(NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN, batch_size=batch_size, allocator=allocator)

    # Prefill
    prompt = mx.random.uniform(0, VOCAB_SIZE, (batch_size, prompt_len)).astype(mx.uint32)
    logits, cache = model(prompt, cache=cache)
    mx.eval(logits)
    cache.advance(prompt_len)
    next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    mx.eval(next_token)

    # Warmup decode
    for _ in range(5):
        logits, cache = model(next_token, cache=cache)
        mx.eval(logits)
        cache.advance(1)
        next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(next_token)

    # Timed decode
    print(f"{'step':>5} {'latency_ms':>11} {'seq_len':>8} {'tok/s':>8}")
    print("-" * 40)

    step_times = []
    for i in range(gen_tokens):
        seq_len = int(prompt_len + 5 + i)  # warmup already advanced 5

        t0 = time.perf_counter()
        logits, cache = model(next_token, cache=cache)
        mx.eval(logits)
        dt = time.perf_counter() - t0

        cache.advance(1)
        next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(next_token)

        step_times.append(dt)
        tok_s = batch_size / dt
        print(f"{i:5d} {dt*1000:11.2f} {seq_len:8d} {tok_s:8.1f}")

    print("-" * 40)
    avg = sum(step_times) / len(step_times)
    print(f"  avg: {avg*1000:.2f} ms/step  ({batch_size/avg:.1f} tok/s)")
    print(f"  min: {min(step_times)*1000:.2f} ms")
    print(f"  max: {max(step_times)*1000:.2f} ms")


def profile_sync():
    """Per-section GPU timings with mx.eval() barriers (serialized, adds overhead)."""
    model = build_our_model()
    mx.eval(model.parameters())

    batch_size = 32
    prompt_len = 32
    gen_tokens = 10

    max_seq = prompt_len + gen_tokens + 10
    num_blocks = (max_seq * batch_size + 127) // 128 + batch_size
    allocator = BlockAllocator(num_blocks, 128, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM)
    cache = KVCache(NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN, batch_size=batch_size, allocator=allocator)

    # Prefill
    prompt = mx.random.uniform(0, VOCAB_SIZE, (batch_size, prompt_len)).astype(mx.uint32)
    logits, cache = model(prompt, cache=cache)
    mx.eval(logits)
    cache.advance(prompt_len)
    next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    mx.eval(next_token)

    # Warmup
    for _ in range(3):
        logits, cache = model(next_token, cache=cache)
        mx.eval(logits)
        cache.advance(1)
        next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(next_token)

    from model import _rope_workaround

    print("\nSynchronized per-section profiling (serialized, expect higher total):\n")

    for step in range(gen_tokens):
        timings = {"embed": 0, "attn_prep": 0, "cache_update": 0, "sdpa": 0, "mlp": 0, "head": 0}

        # Embed
        t0 = time.perf_counter()
        x = model.embed_tokens(next_token)
        mx.eval(x)
        timings["embed"] = time.perf_counter() - t0

        mask = None

        for layer_idx, layer in enumerate(model.layers):
            # Attention prep: norm, qkv proj, reshape, rope
            t0 = time.perf_counter()
            norm_x = layer.input_layernorm(x)
            qkv = layer.self_attn.qkv_proj(norm_x)

            q_size = layer.self_attn.num_heads * layer.self_attn.head_dim
            k_size = layer.self_attn.num_kv_heads * layer.self_attn.head_dim
            q = qkv[..., :q_size]
            k = qkv[..., q_size : q_size + k_size]
            v = qkv[..., q_size + k_size :]

            if layer.self_attn.use_qk_norm:
                q = q.reshape(batch_size, 1, layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(0, 2, 1, 3)
                k = k.reshape(batch_size, 1, layer.self_attn.num_kv_heads, layer.self_attn.head_dim).transpose(
                    0, 2, 1, 3
                )
                q = layer.self_attn.q_norm(q)
                k = layer.self_attn.k_norm(k)
                q = _rope_workaround(
                    q, layer.self_attn.head_dim, layer.self_attn.rope_theta, cache.offsets,
                    traditional=layer.self_attn.rope_traditional,
                )
                k = _rope_workaround(
                    k, layer.self_attn.head_dim, layer.self_attn.rope_theta, cache.offsets,
                    traditional=layer.self_attn.rope_traditional,
                )

            v = v.reshape(batch_size, 1, layer.self_attn.num_kv_heads, layer.self_attn.head_dim).transpose(0, 2, 1, 3)
            mx.eval(q, k, v)
            timings["attn_prep"] += time.perf_counter() - t0

            # Cache update (scatter + gather/concat)
            t0 = time.perf_counter()
            k, v = cache.update(k, v, layer_idx)
            mx.eval(k, v)
            timings["cache_update"] += time.perf_counter() - t0

            # SDPA + output proj
            t0 = time.perf_counter()
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=layer.self_attn.scale, mask=mask)
            out = out.transpose(0, 2, 1, 3).reshape(batch_size, 1, -1)
            attn_out = layer.self_attn.o_proj(out)
            x = x + attn_out
            mx.eval(x)
            timings["sdpa"] += time.perf_counter() - t0

            # MLP
            t0 = time.perf_counter()
            mlp_out = layer.mlp(layer.post_attention_layernorm(x))
            x = x + mlp_out
            mx.eval(x)
            timings["mlp"] += time.perf_counter() - t0

        # Head
        t0 = time.perf_counter()
        x = model.norm(x)
        logits = x @ model.embed_tokens.weight.T
        mx.eval(logits)
        timings["head"] = time.perf_counter() - t0

        cache.advance(1)
        next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(next_token)

        total = sum(timings.values())
        print(f"Step {step}: {total*1000:.1f}ms total")
        for name, dt in timings.items():
            pct = dt / total * 100
            bar = "#" * int(pct / 2)
            print(f"  {name:>14}: {dt*1000:7.2f}ms  ({pct:5.1f}%)  {bar}")
        print()


def profile_gpu_capture():
    """Capture a Metal GPU trace for Xcode Instruments analysis."""
    model = build_our_model()
    mx.eval(model.parameters())

    batch_size = 32
    prompt_len = 32
    gen_tokens = 10

    max_seq = prompt_len + gen_tokens + 10
    num_blocks = (max_seq * batch_size + 127) // 128 + batch_size
    allocator = BlockAllocator(num_blocks, 128, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM)
    cache = KVCache(NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN, batch_size=batch_size, allocator=allocator)

    # Prefill
    prompt = mx.random.uniform(0, VOCAB_SIZE, (batch_size, prompt_len)).astype(mx.uint32)
    logits, cache = model(prompt, cache=cache)
    mx.eval(logits)
    cache.advance(prompt_len)
    next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    mx.eval(next_token)

    # Warmup
    for _ in range(3):
        logits, cache = model(next_token, cache=cache)
        mx.eval(logits)
        cache.advance(1)
        next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(next_token)

    # Capture 3 decode steps
    print("Starting Metal GPU capture (3 steps)...")
    mx.metal.start_capture("profile_decode.gputrace")

    for _ in range(3):
        logits, cache = model(next_token, cache=cache)
        mx.eval(logits)
        cache.advance(1)
        next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(next_token)

    mx.metal.stop_capture()
    print("Saved profile_decode.gputrace — open in Xcode Instruments for kernel-level analysis.")


MODES = {
    "e2e": ("End-to-end step latency (recommended starting point)", profile_e2e),
    "sync": ("Per-section timings with eval barriers (serialized)", profile_sync),
    "gpu": ("Metal GPU trace capture for Xcode Instruments", profile_gpu_capture),
}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "e2e"
    if mode not in MODES:
        print(f"Usage: python profile_decode.py [{' | '.join(MODES)}]")
        for name, (desc, _) in MODES.items():
            print(f"  {name:6s} — {desc}")
        sys.exit(1)

    desc, fn = MODES[mode]
    print(f"Mode: {mode} — {desc}\n")
    fn()
