"""Benchmark prefill and decode performance for Qwen3-0.6B on MLX."""

import time

import mlx.core as mx
import numpy as np
from kvcache import KVCache
from load_weights import download_qwen3, load_qwen3_weights
from model import Qwen3

MODEL_NAME = "Qwen/Qwen3-0.6B"
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

WARMUP_ITERS = 2
BENCH_ITERS = 5
PREFILL_SEQ_LENS = [128, 512, 1024, 2048]
DECODE_TOKENS = 100


def build_model() -> Qwen3:
    print("Downloading/locating checkpoint...")
    checkpoint_path = download_qwen3(MODEL_NAME)
    print(f"Checkpoint at: {checkpoint_path}")

    print("Loading model...")
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
    )
    load_qwen3_weights(model, checkpoint_path)
    return model


def random_tokens(seq_len: int) -> mx.array:
    return mx.array(np.random.randint(0, VOCAB_SIZE, size=(1, seq_len)))


def bench_prefill(model: Qwen3, seq_len: int) -> float:
    tokens = random_tokens(seq_len)

    for _ in range(WARMUP_ITERS):
        logits, _ = model(tokens, cache=None, cur_pos=0)
        mx.eval(logits)

    times: list[float] = []
    for _ in range(BENCH_ITERS):
        t0 = time.perf_counter()
        logits, _ = model(tokens, cache=None, cur_pos=0)
        mx.eval(logits)
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    return seq_len / avg


def bench_decode(model: Qwen3, prompt_len: int = 32, gen_tokens: int = DECODE_TOKENS) -> float:
    prompt = random_tokens(prompt_len)

    cache = KVCache(NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN)
    logits, cache = model(prompt, cache=cache, cur_pos=0)
    mx.eval(logits)
    cache.advance(prompt_len)
    next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)

    # Warmup decode steps
    warmup_cache = KVCache(NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN)
    logits_w, warmup_cache = model(prompt, cache=warmup_cache, cur_pos=0)
    mx.eval(logits_w)
    warmup_cache.advance(prompt_len)
    tok = mx.argmax(logits_w[:, -1, :], axis=-1, keepdims=True)
    for i in range(min(5, gen_tokens)):
        logits_w, warmup_cache = model(tok, cache=warmup_cache, cur_pos=prompt_len + i)
        mx.eval(logits_w)
        warmup_cache.advance(1)
        tok = mx.argmax(logits_w[:, -1, :], axis=-1, keepdims=True)

    # Timed decode
    t0 = time.perf_counter()
    cur_pos = prompt_len
    for _ in range(gen_tokens):
        logits, cache = model(next_token, cache=cache, cur_pos=cur_pos)
        mx.eval(logits)
        cache.advance(1)
        next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        cur_pos += 1
    total = time.perf_counter() - t0

    return gen_tokens / total


def main():
    model = build_model()

    print(f"\n{'=' * 60}")
    print("PREFILL BENCHMARK")
    print(f"{'=' * 60}")
    print(f"  Warmup: {WARMUP_ITERS} iters, Bench: {BENCH_ITERS} iters")
    print(f"{'─' * 60}")
    for seq_len in PREFILL_SEQ_LENS:
        tps = bench_prefill(model, seq_len)
        print(f"  seq_len={seq_len:>5d}  →  {tps:>8.1f} tokens/s")

    print(f"\n{'=' * 60}")
    print("DECODE BENCHMARK")
    print(f"{'=' * 60}")
    print(f"  Prompt: 32 tokens, Generate: {DECODE_TOKENS} tokens")
    print(f"{'─' * 60}")
    tps = bench_decode(model, prompt_len=32, gen_tokens=DECODE_TOKENS)
    print(f"  decode  →  {tps:>8.1f} tokens/s")

    print(f"\n{'=' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()
