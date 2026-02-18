"""Benchmark fused Metal kernels vs separate MLX ops."""

import time

import mlx.core as mx

from model import baseline_norm_rope, baseline_rope, fused_norm_rope, fused_rope

# Nanbeige4.1-3B config
NUM_HEADS = 20
NUM_KV_HEADS = 4
HEAD_DIM = 128
DIM = NUM_HEADS * HEAD_DIM  # 2560 for Q
KV_DIM = NUM_KV_HEADS * HEAD_DIM  # 512 for K
EPS = 1e-5
ROPE_THETA = 70000000.0
ROPE_TRADITIONAL = False

WARMUP = 5
ITERS = 50


def bench(label: str, fn, warmup: int = WARMUP, iters: int = ITERS) -> float:
    for _ in range(warmup):
        result = fn()
        mx.eval(result)

    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        result = fn()
        mx.eval(result)
        times.append(time.perf_counter() - t0)

    avg_us = sum(times) / len(times) * 1e6
    med_us = sorted(times)[len(times) // 2] * 1e6
    print(f"  {label:<30s}  avg {avg_us:>8.1f} µs   med {med_us:>8.1f} µs")
    return avg_us


def correctness_check():
    print("Correctness check...")
    for s in [1, 16, 128, 512]:
        q_proj = mx.random.normal(shape=(1, s, DIM)).astype(mx.bfloat16)
        k_proj = mx.random.normal(shape=(1, s, KV_DIM)).astype(mx.bfloat16)
        q_norm_w = mx.random.normal(shape=(HEAD_DIM,)).astype(mx.float32)
        k_norm_w = mx.random.normal(shape=(HEAD_DIM,)).astype(mx.float32)

        for offset in [0, 7, 100]:
            q_fused = fused_norm_rope(q_proj, q_norm_w, NUM_HEADS, HEAD_DIM, offset, ROPE_THETA)
            q_base = baseline_norm_rope(
                q_proj,
                q_norm_w,
                EPS,
                NUM_HEADS,
                HEAD_DIM,
                offset,
                ROPE_THETA,
                ROPE_TRADITIONAL,
            )
            mx.eval(q_fused, q_base)

            diff = mx.abs(q_fused.astype(mx.float32) - q_base.astype(mx.float32))
            max_diff = mx.max(diff).item()

            k_fused = fused_norm_rope(k_proj, k_norm_w, NUM_KV_HEADS, HEAD_DIM, offset, ROPE_THETA)
            k_base = baseline_norm_rope(
                k_proj,
                k_norm_w,
                EPS,
                NUM_KV_HEADS,
                HEAD_DIM,
                offset,
                ROPE_THETA,
                ROPE_TRADITIONAL,
            )
            mx.eval(k_fused, k_base)

            k_diff = mx.max(mx.abs(k_fused.astype(mx.float32) - k_base.astype(mx.float32))).item()

            status = "PASS" if max_diff < 0.05 and k_diff < 0.05 else "FAIL"
            print(f"  S={s:>4d} offset={offset:>3d}  Q max_diff={max_diff:.4f}  K max_diff={k_diff:.4f}  [{status}]")

            if status == "FAIL":
                print("    CORRECTNESS FAILURE — aborting benchmark")
                return False
    print()
    return True


def rope_correctness_check():
    print("Rope-only correctness check...")
    for s in [1, 16, 128, 512]:
        q_proj = mx.random.normal(shape=(1, s, DIM)).astype(mx.bfloat16)
        k_proj = mx.random.normal(shape=(1, s, KV_DIM)).astype(mx.bfloat16)

        for offset in [0, 7, 100]:
            q_fused = fused_rope(q_proj, NUM_HEADS, HEAD_DIM, offset, ROPE_THETA)
            q_base = baseline_rope(q_proj, NUM_HEADS, HEAD_DIM, offset, ROPE_THETA, ROPE_TRADITIONAL)
            mx.eval(q_fused, q_base)
            max_diff = mx.max(mx.abs(q_fused.astype(mx.float32) - q_base.astype(mx.float32))).item()

            k_fused = fused_rope(k_proj, NUM_KV_HEADS, HEAD_DIM, offset, ROPE_THETA)
            k_base = baseline_rope(k_proj, NUM_KV_HEADS, HEAD_DIM, offset, ROPE_THETA, ROPE_TRADITIONAL)
            mx.eval(k_fused, k_base)
            k_diff = mx.max(mx.abs(k_fused.astype(mx.float32) - k_base.astype(mx.float32))).item()

            status = "PASS" if max_diff < 0.05 and k_diff < 0.05 else "FAIL"
            print(f"  S={s:>4d} offset={offset:>3d}  Q max_diff={max_diff:.4f}  K max_diff={k_diff:.4f}  [{status}]")
            if status == "FAIL":
                print("    CORRECTNESS FAILURE — aborting benchmark")
                return False
    print()
    return True


def run_benchmarks():
    print(f"{'=' * 70}")
    print("FUSED KERNEL BENCHMARKS")
    print(f"  warmup={WARMUP}  iters={ITERS}")
    print(f"  num_heads={NUM_HEADS}  num_kv_heads={NUM_KV_HEADS}  head_dim={HEAD_DIM}")
    print(f"{'=' * 70}\n")

    for s in [1, 16, 128, 512, 1024, 2048]:
        q_proj = mx.random.normal(shape=(1, s, DIM)).astype(mx.bfloat16)
        k_proj = mx.random.normal(shape=(1, s, KV_DIM)).astype(mx.bfloat16)
        q_norm_w = mx.random.normal(shape=(HEAD_DIM,)).astype(mx.float32)
        k_norm_w = mx.random.normal(shape=(HEAD_DIM,)).astype(mx.float32)
        mx.eval(q_proj, k_proj, q_norm_w, k_norm_w)

        print(f"--- B=1, S={s} ---")

        # Rope-only benchmark (Nanbeige / Llama path)
        def baseline_rope_qk():
            q = baseline_rope(q_proj, NUM_HEADS, HEAD_DIM, 0, ROPE_THETA, ROPE_TRADITIONAL)
            k = baseline_rope(k_proj, NUM_KV_HEADS, HEAD_DIM, 0, ROPE_THETA, ROPE_TRADITIONAL)
            return q, k

        def fused_rope_qk():
            q = fused_rope(q_proj, NUM_HEADS, HEAD_DIM, 0, ROPE_THETA)
            k = fused_rope(k_proj, NUM_KV_HEADS, HEAD_DIM, 0, ROPE_THETA)
            return q, k

        t_base = bench("baseline rope (MLX ops)", baseline_rope_qk)
        t_fused = bench("fused rope Metal kernel", fused_rope_qk)
        speedup = t_base / t_fused if t_fused > 0 else float("inf")
        print(f"  {'speedup (rope)':<30s}  {speedup:>8.2f}x")

        # Norm+Rope benchmark (Qwen3 path)
        def baseline_norm_rope_qk():
            q = baseline_norm_rope(q_proj, q_norm_w, EPS, NUM_HEADS, HEAD_DIM, 0, ROPE_THETA, ROPE_TRADITIONAL)
            k = baseline_norm_rope(k_proj, k_norm_w, EPS, NUM_KV_HEADS, HEAD_DIM, 0, ROPE_THETA, ROPE_TRADITIONAL)
            return q, k

        def fused_norm_rope_qk():
            q = fused_norm_rope(q_proj, q_norm_w, NUM_HEADS, HEAD_DIM, 0, ROPE_THETA)
            k = fused_norm_rope(k_proj, k_norm_w, NUM_KV_HEADS, HEAD_DIM, 0, ROPE_THETA)
            return q, k

        t_base = bench("baseline norm+rope (MLX ops)", baseline_norm_rope_qk)
        t_fused = bench("fused norm+rope Metal kernel", fused_norm_rope_qk)
        speedup = t_base / t_fused if t_fused > 0 else float("inf")
        print(f"  {'speedup (norm+rope)':<30s}  {speedup:>8.2f}x\n")


if __name__ == "__main__":
    ok = correctness_check()
    ok = rope_correctness_check() and ok
    if ok:
        run_benchmarks()
