# MLX Inference Node

gRPC server implementation for distributed Nanbeige4.1-3B inference on Apple Silicon.

## Quick Start

### 1. Generate gRPC code (first time only)

```bash
uv run mlx-impl/generate_proto.py
```

### 2. Start the MLX inference node

```bash
# Using default settings (port 50052)
uv run mlx-impl/node.py

# Or with custom options
uv run mlx-impl/node.py --port 50052 --checkpoint Nanbeige/Nanbeige4.1-3B
```

### 3. Test the server (in another terminal)

```bash
uv run mlx-impl/test_grpc_client.py
```

## Files

- **`node.py`** - Main gRPC server implementation for MLX/Nanbeige4.1
- **`generate_proto.py`** - Script to generate Python gRPC code from `.proto` files
- **`test_grpc_client.py`** - Test client for validating the server
- **`generated/`** - Auto-generated gRPC code (don't edit manually)
  - `inference_pb2.py` - Protocol buffer messages
  - `inference_pb2_grpc.py` - gRPC service stubs
  - `inference_pb2.pyi` - Type hints

## Model: Nanbeige4.1-3B

- **Parameters**: 3B
- **Architecture**: Llama-style decoder-only transformer with Grouped Query Attention
- **Layers**: 32 decoder layers
- **Hidden Size**: 2560
- **Vocabulary**: 166,144 tokens
- **Context Length**: 262,144 tokens
- **Optimized for**: Apple Silicon (M1/M2/M3/M4) via MLX framework

## gRPC Service API

The MLX inference node implements the `InferenceNode` gRPC service:

### Prefill

Process input prompt and populate KV cache

- **Input**: `PrefillRequest` (request_id, tokens, temperature, max_tokens)
- **Output**: `PrefillResponse` (next_token_id, cache_position)

### Decode

Generate next token using cached KV state

- **Input**: `DecodeRequest` (request_id, cache_position)
- **Output**: `DecodeResponse` (token_id, is_finished)

### StreamGenerate

Stream generated tokens for a request

- **Input**: `GenerateRequest` (request_id, tokens, temperature, max_tokens)
- **Output**: Stream of `TokenResponse` (token_id, is_finished)

### GetStatus

Get node health and utilization metrics

- **Input**: `StatusRequest` (empty)
- **Output**: `NodeStatus` (active_sequences, gpu_utilization, queue_depth, cached_tokens)

## Architecture

The MLX inference node:

1. Loads Nanbeige4.1-3B model weights from HuggingFace
2. Uses MLX framework for Metal GPU acceleration on Apple Silicon
3. Manages KV cache for efficient autoregressive generation
4. Tracks active sequences for continuous batching
5. Exposes gRPC API for the scheduler

See `../INFERENCE_SERVER.md` for the full distributed architecture.

## Configuration

Command-line options:

- `--port` - gRPC server port (default: 50052)
- `--checkpoint` - HuggingFace model path (default: Nanbeige/Nanbeige4.1-3B)
- `--max-batch-size` - Max concurrent sequences (default: 32)

## Performance

Based on benchmarks (see `MLX.md`):

**Prefill** (prompt processing):

- 128 tokens: 1,100.5 tokens/s
- 512 tokens: 1,141.7 tokens/s
- 1024 tokens: 1,101.9 tokens/s
- 2048 tokens: 1,056.9 tokens/s

**Decode** (autoregressive generation):

- Single-batch: 36.3 tokens/s (memory bandwidth limited)

MLX uses optimized Metal kernels (`mx.fast.*`) for:

- RoPE (Rotary Position Embeddings)
- Scaled Dot-Product Attention
- RMSNorm

## Memory Requirements

- **Model weights**: ~1.2 GB (bfloat16)
- **With activations**: ~2-3 GB total
- **Recommended**: 8GB+ unified memory (M1/M2/M3 base models work fine)

## Development

Regenerate gRPC code after changing `scheduler/proto/inference.proto`:

```bash
uv run mlx-impl/generate_proto.py
```

Run linting:

```bash
uvx ruff check mlx-impl/node.py
uvx ruff format mlx-impl/node.py
```

Type checking:

```bash
uvx pyright mlx-impl/node.py
```

## Next Steps

- [ ] Implement continuous batching (dynamic batch management)
- [ ] Add cache persistence between requests (currently recreated)
- [ ] Add Metal GPU utilization monitoring
- [ ] Add Prometheus metrics export
- [ ] Integrate with Rust scheduler
- [ ] Add health check heartbeat to scheduler
- [ ] Optimize cache storage strategy

## Related Documentation

- `MLX.md` - Implementation details and benchmarks
- `../INFERENCE_SERVER.md` - Distributed inference architecture
