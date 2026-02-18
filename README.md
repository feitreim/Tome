# Tome

MLX implementation of Nanbeige4.1-3B, a dense decoder-only transformer with Grouped Query Attention, optimized for Apple Silicon.

## Installation

```bash
uv sync
```

## Running

Start the scheduler, inference node, and TUI in three terminals:

```bash
# Terminal 1: Scheduler (Rust HTTP server, routes requests to nodes)
./start_scheduler.sh

# Terminal 2: MLX inference node (gRPC server, runs the model)
./start_node.sh

# Terminal 3: TUI (interactive chat interface)
./start_tui.sh
```

Register the node with the scheduler (You shouldn't need to do this):

```bash
curl -X POST http://localhost:8080/v1/nodes \
  -H "Content-Type: application/json" \
  -d '{"addr": "http://localhost:50052"}'
```

### Tests & Benchmarks

```bash
# Run component tests against HuggingFace reference
uv run mlx-impl/test_components.py

# Measure KL divergence vs HuggingFace
uv run mlx-impl/measure_kl_div.py

# Benchmark inference
uv run mlx-impl/benchmark.py
```

First run will download the Nanbeige4.1-3B checkpoint from HuggingFace.

## OpenCode

To use Tome as a backend for [OpenCode](https://opencode.ai), add this to `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "tome": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Tome (local)",
      "options": {
        "baseURL": "http://localhost:8080/v1"
      },
      "models": {
        "Nanbeige/Nanbeige4.1-3B": {
          "name": "Nanbeige4.1-3B (Tome MLX)",
          "limit": {
            "context": 262144,
            "output": 4096
          }
        }
      }
    }
  }
}
```

Then start the scheduler and node, and select the Tome provider in OpenCode.

## Project Structure

- `mlx-impl/model.py` - Nanbeige4.1-3B model implementation using MLX
- `mlx-impl/load_weights.py` - HuggingFace checkpoint loading
- `mlx-impl/kvcache.py` - KV cache for autoregressive generation
- `mlx-impl/node.py` - gRPC inference node server
- `mlx-impl/test_components.py` - Validation tests vs HuggingFace Transformers
- `mlx-impl/measure_kl_div.py` - KL divergence measurement
- `mlx-impl/benchmark.py` - Inference benchmarks
- `mlx-impl/benchmark_kernel.py` - Metal kernel microbenchmarks
- `scheduler/` - Rust inference scheduler
- `tui/` - Ratatui interactive chat interface

## Development

```bash
# Lint
uvx ruff check .

# Format
uvx ruff format .

# Type check
uvx ty check
```
