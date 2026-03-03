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
./start_scheduler.sh

# Terminal 2: MLX inference node (gRPC server, runs Qwen3-0.6B by default)
./start_node.sh

# Terminal 3: TUI (interactive chat interface)
./start_tui.sh
```

### GRPO Training iteration (example)

```bash
curl -X POST http://localhost:8080/v1/grpo \
  -H "Content-Type: application/json" \
  -d '{
    "batch_id": "iter-1",
    "prompts": [{"prompt_id": "p1", "prompt": "Write a poem about tensors."}],
    "group_size": 8,
    "judge": {"rubric": "Technical accuracy and rhyme.", "temperature": 0.0, "max_tokens": 1}
  }'
```

### Tests & Benchmarks

```bash
# Run paged KV and CoW tests
uv run python3 mlx-impl/test_kvcache_batched.py

# Run full GRPO pipeline test
uv run python3 mlx-impl/test_grpo_rpc.py

# Benchmark inference (defaults to Qwen3-0.6B)
uv run python3 mlx-impl/benchmark.py
```

## Project Structure

- `mlx-impl/node.py` - gRPC inference node server with GRPO support.
- `mlx-impl/kvcache.py` - Paged KV cache with CoW and Prefix Cache.
- `mlx-impl/model.py` - Qwen3/Nanbeige model implementation in MLX.
- `scheduler/` - Rust inference scheduler and prefix-aware router.
- `tui/` - Ratatui interactive chat interface.

## Documentation

- [Inference Server Architecture](INFERENCE_SERVER.md)
- [MLX Implementation Details](mlx-impl/README.md)
- [Scheduler API & Routing](scheduler/SCHEDULER.md)
