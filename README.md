# Tome

Distributed LLM inference and **GRPO training infrastructure** for Apple Silicon, featuring Paged KV Cache and Prefix Caching.

## Features

- **Optimized for GRPO**: Unified rollout-judge-train pipeline with multi-level prefix caching.
- **Paged KV Cache**: 128-token block-based memory management with Copy-on-Write (CoW).
- **Prefix Caching**: Radix-tree based KV reuse for shared rubrics and prompts.
- **Dual Model Architecture**: Policy and Reference models live in unified memory for instant log-prob computation.
- **In-place Weight Updates**: Merge LoRA adapters via RPC without reloads.
- **Distributed Scheduler**: Rust-based orchestrator with prefix-aware routing.

## Installation

```bash
uv sync
```

## Running

Start the scheduler, inference node, and TUI in three terminals:

```bash
# Terminal 1: Scheduler (Rust HTTP server, routes requests to nodes)
./scripts/start_scheduler.sh

# Terminal 2: MLX inference node (gRPC server, runs Qwen3-0.6B by default)
./scripts/start_node.sh

# Terminal 3: TUI (interactive chat interface)
./scripts/start_tui.sh
```

### GRPO Training iteration (example)

The GRPO API is split into **Rollout** (generates completions + ref logprobs) and **Judge** (scores completions):

```bash
# 1. Rollout
curl -X POST http://localhost:8080/v1/grpo/rollout \
  -H "Content-Type: application/json" \
  -d '{
    "batch_id": "iter-1",
    "prompts": [{"prompt_id": "p1", "prompt": "Write a poem about tensors."}],
    "group_size": 8,
    "max_tokens": 128
  }'

# 2. Judge
curl -X POST http://localhost:8080/v1/grpo/judge \
  -H "Content-Type: application/json" \
  -d '{
    "batch_id": "iter-1",
    "rubric": "Technical accuracy and rhyme.",
    "items": [{"item_id": "i1", "prompt": "Prompt: Poem about tensors. Completion: ..."}]
  }'
```

### Tests & Benchmarks

```bash
# Run End-to-End GRPO pipeline test and benchmark
uv run tests/e2e_grpo.py

# Expected Output (approximate):
# --- BENCHMARK ---
# rollout + ref_lps Throughput: 136.0 tokens/s (aggregate)
# Judge Throughput: 4.3 items/s
uv run python3 mlx-impl/tests/test_kvcache.py
uv run python3 mlx-impl/tests/test_grpo.py

# Benchmark kernel and model performance
uv run python3 mlx-impl/benchmarks/benchmark_kernel.py
uv run python3 mlx-impl/benchmarks/benchmark.py
```

## Project Structure

- `mlx-impl/node.py` - gRPC inference node server with GRPO support.
- `mlx-impl/kvcache.py` - Paged KV cache with CoW and Prefix Cache.
- `mlx-impl/model.py` - Qwen3/Nanbeige model implementation in MLX.
- `mlx-impl/tests/` - Unit tests for MLX components.
- `mlx-impl/benchmarks/` - Performance and comparison benchmarks.
- `scheduler/` - Rust inference scheduler and prefix-aware router.
- `tests/e2e_grpo.py` - System-wide end-to-end integration test.
- `scripts/` - Convenience scripts for starting components.
- `tui/` - Ratatui interactive chat interface.

## Documentation

- [Inference Server Architecture](INFERENCE_SERVER.md)
- [MLX Implementation Details](mlx-impl/README.md)
- [Scheduler API & Routing](scheduler/SCHEDULER.md)
