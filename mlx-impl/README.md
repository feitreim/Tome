# MLX Inference Node

gRPC server implementation for distributed LLM inference on Apple Silicon, optimized for **GRPO training loops**.

## Quick Start

### 1. Generate gRPC code (first time only)

```bash
uv run mlx-impl/generate_proto.py
```

### 2. Start the MLX inference node

```bash
# Using default settings (port 50052, Qwen3-0.6B)
uv run mlx-impl/node.py

# Or with custom options
uv run mlx-impl/node.py --port 50052 --checkpoint Qwen/Qwen3-0.6B
```

### 3. Test the server (in another terminal)

```bash
uv run python3 mlx-impl/test_grpc_client.py
```

## Features

- **Paged KV Cache**: Efficient memory management with 128-token blocks.
- **Copy-on-Write (CoW)**: Zero-copy sharing of KV prefixes between sequences (critical for GRPO group rollouts).
- **Prefix Caching**: Automatic radix-tree based caching of common prefixes (prompts, rubrics).
- **Dual Model Setup**: Maintains a Policy model (trainable via LoRA) and a frozen Reference model in unified memory.
- **In-place LoRA Merging**: Update policy weights without model reloads or server restarts.

## Model: Qwen3-0.6B (Default)

- **Parameters**: 0.6B
- **Architecture**: Qwen-style transformer with QK-Norm and Grouped Query Attention.
- **Context Length**: 40,960 tokens
- **Memory Footprint**: ~1.2 GB (weights) + KV Cache.
- **Optimized for**: Apple Silicon (M1/M2/M3/M4) via MLX framework.

## gRPC Service API

The MLX inference node implements the `InferenceNode` gRPC service:

### GRPO

Full rollout + judge + reference log-prob pipeline.
- **Rollouts**: Generates $G$ completions per prompt.
- **Judge**: Scores completions using a shared rubric and 3-level prefix caching.
- **Ref Log-probs**: Computes log-probabilities using the frozen reference model.

### UpdateWeights

Merge LoRA adapters into the active policy model.
- **Input**: Layer updates with base64-encoded $A$ and $B$ matrices.

### StreamGenerate / Prefill / Decode

Standard inference endpoints for single-sequence and batched generation.

### GetStatus

Get node health, active sequences, and KV cache utilization.

## Performance (Qwen3-0.6B)

**Prefill** (batch_size=1):
- ~600 tokens/s

**Decode** (aggregate throughput):
- batch_size=1: ~9 tokens/s
- batch_size=32: ~150 tokens/s
- batch_size=64: ~190 tokens/s

## Memory Requirements

- **Base Models**: Works on 8GB machines.
- **Dual Model Setup**: ~2.5 GB for weights + KV Cache. 16GB+ recommended for large batch sizes.

## Development

Regenerate gRPC code:
```bash
uv run mlx-impl/generate_proto.py
```

Run specialized tests:
- `test_kvcache_batched.py`: Paged KV correctness.
- `test_prefix_caching.py`: Prefix hit/miss and CoW verification.
- `test_weight_update.py`: LoRA merge accuracy.
- `test_grpo_rpc.py`: Full end-to-end training pipeline.
