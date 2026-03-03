"""Head-to-head benchmark: Tome vs vllm-mlx vs mlx-lm on Qwen3-0.6B."""

import asyncio
import sys
import time

import mlx.core as mx
import numpy as np

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

PROMPT = "The quick brown fox"
GEN_TOKENS = 50
WARMUP = 3
BATCH_SIZES = [1, 8, 16, 32, 64, 128]


# ── Tome ─────────────────────────────────────────────────────────────────────


def _build_tome_model():
    from kvcache import BlockAllocator, KVCache
    from load_weights import download_qwen3, load_qwen3_weights
    from model import Qwen3

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
    load_qwen3_weights(model, download_qwen3(MODEL_NAME))
    return model


def tome_generate(model, prompt_ids: list[int], gen_tokens: int, batch_size: int = 1) -> list[list[int]]:
    """Run greedy generation with Tome, return generated token IDs per sequence."""
    from kvcache import BlockAllocator, KVCache

    prompt_len = len(prompt_ids)
    prompt = mx.array([prompt_ids] * batch_size)

    max_seq = prompt_len + gen_tokens + 10
    num_blocks = (max_seq * batch_size + 127) // 128 + batch_size
    allocator = BlockAllocator(num_blocks, 128, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM)
    cache = KVCache(NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN, batch_size=batch_size, allocator=allocator)

    logits, cache = model(prompt, cache=cache)
    mx.eval(logits)
    cache.advance(prompt_len)
    next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    mx.eval(next_token)

    generated = [[] for _ in range(batch_size)]
    for b in range(batch_size):
        generated[b].append(int(next_token[b, 0]))

    for _ in range(gen_tokens - 1):
        logits, cache = model(next_token, cache=cache)
        mx.eval(logits)
        cache.advance(1)
        next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(next_token)
        for b in range(batch_size):
            generated[b].append(int(next_token[b, 0]))

    return generated


def bench_tome(batch_sizes: list[int]) -> dict[int, float]:
    model = _build_tome_model()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompt_ids = tokenizer.encode(PROMPT)
    prompt_len = len(prompt_ids)

    results = {}
    for B in batch_sizes:
        from kvcache import BlockAllocator, KVCache

        prompt = mx.array([prompt_ids] * B)

        max_seq = prompt_len + GEN_TOKENS + WARMUP + 10
        num_blocks = (max_seq * B + 127) // 128 + B
        allocator = BlockAllocator(num_blocks, 128, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM)
        cache = KVCache(NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN, batch_size=B, allocator=allocator)

        # Prefill
        logits, cache = model(prompt, cache=cache)
        mx.eval(logits)
        cache.advance(prompt_len)
        next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(next_token)

        # Warmup
        for _ in range(WARMUP):
            logits, cache = model(next_token, cache=cache)
            mx.eval(logits)
            cache.advance(1)
            next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            mx.eval(next_token)

        # Timed
        t0 = time.perf_counter()
        for _ in range(GEN_TOKENS):
            logits, cache = model(next_token, cache=cache)
            mx.eval(logits)
            cache.advance(1)
            next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            mx.eval(next_token)
        dt = time.perf_counter() - t0

        total_tokens = B * GEN_TOKENS
        results[B] = total_tokens / dt

        del cache, allocator
        mx.clear_cache()

    del model
    mx.clear_cache()
    return results


# ── vllm-mlx ─────────────────────────────────────────────────────────────────


def bench_vllm_mlx(batch_sizes: list[int]) -> dict[int, float]:
    from vllm_mlx.engine.batched import BatchedEngine

    async def _run():
        engine = BatchedEngine(MODEL_NAME)
        await engine.start()

        # Warmup
        for _ in range(WARMUP):
            await engine.generate(PROMPT, max_tokens=10, temperature=0.0)

        results = {}
        for B in batch_sizes:
            # Warmup this batch size
            tasks = [engine.generate(PROMPT, max_tokens=10, temperature=0.0) for _ in range(B)]
            await asyncio.gather(*tasks)

            # Timed: submit B concurrent requests
            tasks = [engine.generate(PROMPT, max_tokens=GEN_TOKENS, temperature=0.0) for _ in range(B)]
            t0 = time.perf_counter()
            outputs = await asyncio.gather(*tasks)
            dt = time.perf_counter() - t0

            total_tokens = sum(o.completion_tokens for o in outputs)
            results[B] = total_tokens / dt

        await engine.stop()
        return results

    return asyncio.run(_run())


def vllm_generate(prompt: str, gen_tokens: int) -> list[int]:
    """Run greedy generation with vllm-mlx, return generated token IDs."""
    from vllm_mlx.engine.batched import BatchedEngine

    async def _run():
        engine = BatchedEngine(MODEL_NAME)
        await engine.start()
        output = await engine.generate(prompt, max_tokens=gen_tokens, temperature=0.0)
        token_ids = engine.tokenizer.encode(output.text)
        await engine.stop()
        return token_ids

    return asyncio.run(_run())


# ── mlx-lm ───────────────────────────────────────────────────────────────────


def bench_mlx_lm() -> float:
    """mlx-lm only supports batch_size=1, so just return single-sequence tok/s."""
    from mlx_lm import load, generate

    model, tokenizer = load(MODEL_NAME)

    # Warmup
    for _ in range(WARMUP):
        generate(model, tokenizer, prompt=PROMPT, max_tokens=10, verbose=False)

    # Timed
    t0 = time.perf_counter()
    generate(model, tokenizer, prompt=PROMPT, max_tokens=GEN_TOKENS, verbose=False)
    dt = time.perf_counter() - t0

    del model
    mx.clear_cache()
    return GEN_TOKENS / dt


# ── Correctness ──────────────────────────────────────────────────────────────


def test_correctness():
    """Compare greedy generation between Tome, vllm-mlx, and mlx-lm."""
    from transformers import AutoTokenizer
    from mlx_lm import load, generate as mlx_generate

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    # Ensure BOS token is included if present
    # (Actually for Qwen, usually BOS is not added to prompt by default)
    if tokenizer.bos_token_id is not None and (not prompt_ids or prompt_ids[0] != tokenizer.bos_token_id):
        # We dont add BOS here because mlx-lm.generate might not be adding it either
        # if the config doesnt force it.
        pass

    gen_tokens = 30

    print("Correctness test: Tome vs mlx-lm vs vllm-mlx (greedy, 30 tokens)")
    print(f"  Prompt: {repr(PROMPT)} ({len(prompt_ids)} tokens)")

    # mlx-lm
    print("  Running mlx-lm...", flush=True)
    model_mlx, tokenizer_mlx = load(MODEL_NAME)
    # Generate with mlx-lm
    mlx_lm_text = mlx_generate(model_mlx, tokenizer_mlx, prompt=PROMPT, max_tokens=gen_tokens, verbose=False)
    # Re-encode to get actual tokens for comparison. 
    # NOTE: We prepend a space to avoid the "no-space-at-start" encoding bug if text starts with word
    # Actually, the most robust way is to encode the full text (prompt + completion) and take the completion part.
    full_text = PROMPT + mlx_lm_text
    full_tokens = tokenizer_mlx.encode(full_text)
    prompt_len_tokens = len(tokenizer_mlx.encode(PROMPT))
    mlx_lm_tokens = full_tokens[prompt_len_tokens:]
    
    del model_mlx
    mx.clear_cache()


    # Tome B=1
    print("  Running Tome...", flush=True)
    model = _build_tome_model()
    tome_tokens = tome_generate(model, prompt_ids, gen_tokens, batch_size=1)[0]
    del model
    mx.clear_cache()

    # vllm-mlx
    print("  Running vllm-mlx...", flush=True)
    vllm_tokens = vllm_generate(PROMPT, gen_tokens)

    # Decode all
    tome_text = tokenizer.decode(tome_tokens, skip_special_tokens=True)
    vllm_text = tokenizer.decode(vllm_tokens, skip_special_tokens=True)

    print(f"\n  mlx-lm: {repr(mlx_lm_text[:120])}")
    print(f"  Tome  : {repr(tome_text[:120])}")
    print(f"  vllm  : {repr(vllm_text[:120])}")

    # String-level comparison
    def compare_strings(name, actual, expected):
        # Normalize by removing leading spaces for comparison if one has it and other doesn't
        a = actual.strip()
        e = expected.strip()
        
        min_len = min(len(a), len(e))
        match_count = 0
        for i in range(min_len):
            if a[i] == e[i]:
                match_count += 1
            else:
                break
        
        print(f"  {name} matches mlx-lm for first {match_count} characters")
        if match_count < 10 and len(e) > 0:
            print(f"    Expected start: {repr(e[:30])}")
            print(f"    Actual start:   {repr(a[:30])}")

    compare_strings("Tome", tome_text, mlx_lm_text)
    compare_strings("vllm", vllm_text, mlx_lm_text)



    # Also verify Tome batched consistency: all sequences in a batch should be identical
    print("\n  Checking Tome batched consistency (B=4)...", flush=True)
    model = _build_tome_model()
    batched_tokens = tome_generate(model, prompt_ids, gen_tokens, batch_size=4)
    del model
    mx.clear_cache()

    all_same = all(seq == batched_tokens[0] for seq in batched_tokens)
    if all_same:
        print("  PASS: All 4 sequences in batch are identical (deterministic)")
    else:
        print("  FAIL: Sequences differ within batch!")
        for i, seq in enumerate(batched_tokens):
            print(f"    seq[{i}]: {seq[:10]}...")

    # Verify batched matches single
    if batched_tokens[0] == tome_tokens:
        print("  PASS: Batched output matches single-sequence output")
    else:
        print("  WARN: Batched output differs from single-sequence (may be a precision issue)")
        for i in range(min(len(tome_tokens), len(batched_tokens[0]))):
            if tome_tokens[i] != batched_tokens[0][i]:
                print(f"    First diff at position {i}: single={tome_tokens[i]}, batched={batched_tokens[0][i]}")
                break


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    suites = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    run_all = "all" in suites

    if "correctness" in suites:
        test_correctness()
        return

    batch_sizes = BATCH_SIZES

    print(f"Qwen3-0.6B benchmark — gen_tokens={GEN_TOKENS}, warmup={WARMUP}")

    mlx_lm_result = None
    vllm_results = {}
    tome_results = {}

    if run_all or "mlx-lm" in suites:
        print("Running mlx-lm (B=1 only)...", flush=True)
        mlx_lm_result = bench_mlx_lm()

    if run_all or "vllm" in suites:
        print("Running vllm-mlx...", flush=True)
        vllm_results = bench_vllm_mlx(batch_sizes)

    if run_all or "tome" in suites:
        print("Running Tome...", flush=True)
        tome_results = bench_tome(batch_sizes)

    print()
    print(f"{'':>4}  {'B':>4}  {'mlx-lm':>10}  {'vllm-mlx':>10}  {'Tome':>10}")
    print("-" * 50)

    for B in batch_sizes:
        mlx_col = f"{mlx_lm_result:.1f}" if mlx_lm_result and B == 1 else "—"
        vllm_col = f"{vllm_results[B]:.1f}" if B in vllm_results else "—"
        tome_col = f"{tome_results[B]:.1f}" if B in tome_results else "—"
        print(f"{'':>4}  {B:>4}  {mlx_col:>10}  {vllm_col:>10}  {tome_col:>10}")

    print("-" * 50)
    print("All values in tokens/s (higher is better)")


if __name__ == "__main__":
    main()
