# JAX Inference Node

gRPC server implementation for distributed OLMoE inference.

## Quick Start

### 1. Generate gRPC code (first time only)

```bash
uv run jax-impl/generate_proto.py
```

### 2. Start the inference node

```bash
# Using the convenience script
./start_node.sh

# Or directly with options
uv run jax-impl/node.py --port 50051 --num-devices 1
```

### 3. Test the server (in another terminal)

```bash
uv run jax-impl/test_grpc_client.py
```

## Files

- **`node.py`** - Main gRPC server implementation
- **`generate_proto.py`** - Script to generate Python gRPC code from `.proto` files
- **`test_grpc_client.py`** - Test client for validating the server
- **`generated/`** - Auto-generated gRPC code (don't edit manually)
  - `inference_pb2.py` - Protocol buffer messages
  - `inference_pb2_grpc.py` - gRPC service stubs
  - `inference_pb2.pyi` - Type hints

## gRPC Service API

The inference node implements the `InferenceNode` gRPC service with these methods:

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

The inference node:
1. Loads OLMoE model weights (supports tensor parallelism)
2. Manages a KV cache for fast autoregressive generation
3. Tracks active sequences for continuous batching
4. Exposes gRPC API for the scheduler to send requests

See `../INFERENCE_SERVER.md` for the full distributed architecture.

## Configuration

Command-line options:
- `--port` - gRPC server port (default: 50051)
- `--checkpoint` - HuggingFace model path (default: allenai/OLMoE-1B-7B-0924)
- `--num-devices` - GPUs for tensor parallelism (default: 1)
- `--max-batch-size` - Max concurrent sequences (default: 32)

## Tensor Parallelism

For multi-GPU inference:

```bash
# 2 GPUs
uv run jax-impl/node.py --num-devices 2

# 4 GPUs
uv run jax-impl/node.py --num-devices 4
```

Expert weights are automatically sharded across devices.

## Development

Regenerate gRPC code after changing `scheduler/proto/inference.proto`:

```bash
uv run jax-impl/generate_proto.py
```

Run linting:

```bash
uvx ruff check jax-impl/node.py
uvx ruff format jax-impl/node.py
```

Type checking:

```bash
uvx pyright jax-impl/node.py
```

## Next Steps

- [ ] Implement continuous batching (dynamic batch management)
- [ ] Add cache state tracking for prefix-aware routing
- [ ] Implement proper GPU utilization monitoring
- [ ] Add Prometheus metrics export
- [ ] Integrate with Rust scheduler
- [ ] Add health check heartbeat to scheduler
