"""
Benchmark our optimized implementation against vllm-mlx.
Supports both Nanbeige 3B and Qwen 0.6B.
"""

import argparse
import statistics
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm
import numpy as np
from transformers import AutoTokenizer

from kvcache import KVCache
from load_weights import download_qwen3, load_qwen3_weights
from model import Qwen3

try:
    from vllm_mlx.engine_core import EngineConfig, EngineCore
    from vllm_mlx.request import SamplingParams
    from vllm_mlx.scheduler import SchedulerConfig
except ImportError as e:
    print("vllm-mlx is not installed. Install with:")
    print("  uv pip install git+https://github.com/waybarrios/vllm-mlx.git")
    exit(1)

CONFIGS = {
    "Nanbeige/Nanbeige4.1-3B": {
        "vocab_size": 166144,
        "dim": 2560,
        "num_layers": 32,
        "num_heads": 20,
        "num_kv_heads": 4,
        "head_dim": 128,
        "intermediate_size": 10496,
        "max_seq_len": 262144,
        "rope_theta": 70000000.0,
        "eps": 1e-5,
        "tie_word_embeddings": False,
        "use_qk_norm": False,
        "rope_traditional": True,
    },
    "Qwen/Qwen3-0.6B": {
        "vocab_size": 151936,
        "dim": 1024,
        "num_layers": 28,
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 3072,
        "max_seq_len": 40960,
        "rope_theta": 1000000.0,
        "eps": 1e-6,
        "tie_word_embeddings": True,
        "use_qk_norm": True,
        "rope_traditional": False,
    },
}


def build_our_model(model_name: str, checkpoint_path: str):
    config = CONFIGS[model_name]
    model = Qwen3(**config)
    load_qwen3_weights(model, checkpoint_path)
    return model


def our_generate(model, config, tokens, max_tokens, temperature=0.0):
    b, s = tokens.shape
    cache = KVCache(
        config["num_layers"],
        config["num_kv_heads"],
        config["head_dim"],
        config["max_seq_len"],
        batch_size=b,
    )
    
    # Prefill
    logits, cache = model(tokens, cache=cache)
    mx.eval(logits)
    cache.advance(s)
    
    next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    generated = [next_token]
    
    # Decode
    for _ in range(max_tokens - 1):
        logits, cache = model(next_token, cache=cache)
        mx.eval(logits)
        cache.advance(1)
        next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        generated.append(next_token)
        
    return mx.concatenate(generated, axis=1)


def _stats(name: str, times_s: list[float], total_tokens: int) -> None:
    mean_s = statistics.mean(times_s)
    std_s = statistics.pstdev(times_s) if len(times_s) > 1 else 0.0
    tps = total_tokens / sum(times_s) if times_s else 0.0
    print(f"{name:<10} | mean {mean_s * 1000:8.1f} ms | std {std_s * 1000:6.1f} ms | throughput {tps:7.2f} tok/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", choices=list(CONFIGS.keys()))
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    config = CONFIGS[args.model]
    checkpoint_path = download_qwen3(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Use a dummy prompt
    prompt = "What is the capital of France?"
    input_ids = tokenizer.encode(prompt)
    tokens = mx.array([input_ids] * args.batch)
    vllm_input_ids = [input_ids] * args.batch

    print(f"\nConfig: Model={args.model}, B={args.batch}, max_tokens={args.max_tokens}, runs={args.runs}\n")

    # Benchmark Ours
    print("Loading our model...")
    our_model = build_our_model(args.model, checkpoint_path)
    mx.eval(our_model.parameters())
    
    our_times = []
    
    print(f"Ours warmup ({args.warmup})...")
    for _ in range(args.warmup):
        _ = our_generate(our_model, config, tokens, args.max_tokens)

    for i in range(args.runs):
        print(f"Ours run {i+1}...")
        t0 = time.perf_counter()
        _ = our_generate(our_model, config, tokens, args.max_tokens)
        our_times.append(time.perf_counter() - t0)
        
    # Free memory
    del our_model
    mx.clear_cache()
    time.sleep(1)
    
    # Benchmark vLLM
    print("\nLoading vllm-mlx model...")
    vllm_model, vllm_tok = mlx_lm.load(args.model)
    vllm_sched = SchedulerConfig(
        max_num_seqs=args.batch,
        prefill_batch_size=args.batch,
        completion_batch_size=args.batch,
    )
    vllm_engine = EngineCore(
        vllm_model,
        vllm_tok,
        EngineConfig(model_name=args.model, scheduler_config=vllm_sched),
    )
    vllm_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
    )

    vllm_times = []
    
    print(f"vLLM warmup ({args.warmup})...")
    for _ in range(args.warmup):
        _ = vllm_engine.generate_batch_sync(vllm_input_ids, vllm_params)

    for i in range(args.runs):
        print(f"vLLM run {i+1}...")
        t0 = time.perf_counter()
        _ = vllm_engine.generate_batch_sync(vllm_input_ids, vllm_params)
        vllm_times.append(time.perf_counter() - t0)

    total_tokens_per_run = args.batch * args.max_tokens
    print("\n" + "="*50)
    _stats("ours", our_times, total_tokens_per_run * args.runs)
    _stats("vllm_mlx", vllm_times, total_tokens_per_run * args.runs)

    mean_ours = statistics.mean(our_times)
    mean_vllm = statistics.mean(vllm_times)
    speedup = mean_vllm / mean_ours if mean_ours > 0 else 0
    print(f"\nResult: ours is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than vllm-mlx")
    print("="*50)
    
    vllm_engine.close()


if __name__ == "__main__":
    main()
