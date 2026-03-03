# gRPC Inference Node Setup

This guide explains how to set up and run the JAX inference node with gRPC server.

## Prerequisites

1. **Install dependencies**:
   ```bash
   uv sync
   ```

   For CUDA support:
   ```bash
   uv sync --extra cuda12  # or cuda11
   ```

2. **Generate gRPC Python code from protobuf**:
   ```bash
   uv run jax-impl/generate_proto.py
   ```

   This creates Python files in `jax-impl/generated/`:
   - `inference_pb2.py` - Protocol buffer message classes
   - `inference_pb2_grpc.py` - gRPC service stubs
   - `inference_pb2.pyi` - Type stubs for better IDE support

## Running the Server

### Basic Usage

Start the inference node on default port 50051:

```bash
uv run jax-impl/node.py
```

### With Custom Options

```bash
# Custom port
uv run jax-impl/node.py --port 50052

# With tensor parallelism (2 GPUs)
uv run jax-impl/node.py --num-devices 2

# Custom checkpoint
uv run jax-impl/node.py --checkpoint allenai/OLMoE-1B-7B-0924

# All options
uv run jax-impl/node.py \
  --port 50051 \
  --checkpoint allenai/OLMoE-1B-7B-0924 \
  --num-devices 2 \
  --max-batch-size 32
```

### Command-line Options

- `--port`: gRPC server port (default: 50051)
- `--checkpoint`: HuggingFace model checkpoint (default: allenai/OLMoE-1B-7B-0924)
- `--num-devices`: Number of devices for tensor parallelism (default: 1)
- `--max-batch-size`: Maximum batch size for continuous batching (default: 32)

## Testing the Server

### Using the Test Client

In a separate terminal, run the test client:

```bash
uv run jax-impl/test_grpc_client.py
```

This will test all gRPC endpoints:
- `GetStatus` - Get node status
- `Prefill` - Process prompt and populate KV cache
- `Decode` - Generate next token
- `StreamGenerate` - Stream generation tokens

### Using grpcurl (Command-line Tool)

Install grpcurl:
```bash
brew install grpcurl  # macOS
# or
go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest
```

List available services:
```bash
grpcurl -plaintext localhost:50051 list
```

Get node status:
```bash
grpcurl -plaintext localhost:50051 inference.InferenceNode/GetStatus
```

Send prefill request:
```bash
grpcurl -plaintext -d '{
  "request_id": "test-001",
  "tokens": [1, 2, 3, 4, 5],
  "temperature": 0.7,
  "max_tokens": 100
}' localhost:50051 inference.InferenceNode/Prefill
```

## Architecture

### Protocol Buffers Schema

The gRPC interface is defined in `scheduler/proto/inference.proto` with these RPCs:

1. **Prefill** - Process input tokens and populate KV cache
   - Input: `PrefillRequest` (request_id, tokens, temperature, max_tokens)
   - Output: `PrefillResponse` (next_token_id, cache_position)

2. **Decode** - Generate next token using cached KV
   - Input: `DecodeRequest` (request_id, cache_position)
   - Output: `DecodeResponse` (token_id, is_finished)

3. **StreamGenerate** - Stream generated tokens
   - Input: `GenerateRequest` (request_id, tokens, temperature, max_tokens)
   - Output: Stream of `TokenResponse` (token_id, is_finished)

4. **GetStatus** - Get node health and utilization
   - Input: `StatusRequest` (empty)
   - Output: `NodeStatus` (active_sequences, gpu_utilization, queue_depth, cached_tokens)

### Implementation Details

**File**: `jax-impl/node.py`

- `InferenceNodeServicer` - Implements gRPC service
  - Manages KV cache using `kvcache.py`
  - Tracks active sequences for continuous batching
  - Supports temperature-based sampling

- `serve()` - Async server entry point
  - Loads OLMoE model with optional tensor parallelism
  - Starts gRPC server on specified port

## Tensor Parallelism

For models that don't fit in single GPU memory:

```bash
# Use 2 GPUs
uv run jax-impl/node.py --num-devices 2

# Use 4 GPUs
uv run jax-impl/node.py --num-devices 4
```

Expert weights are automatically sharded across devices. See `CLAUDE.md` for memory requirements.

## Troubleshooting

### Proto files not found

If you get import errors for `inference_pb2`, make sure you generated the proto files:
```bash
uv run jax-impl/generate_proto.py
```

### Port already in use

Change the port:
```bash
uv run jax-impl/node.py --port 50052
```

### Out of memory

Use tensor parallelism with multiple GPUs:
```bash
uv run jax-impl/node.py --num-devices 2
```

Or reduce batch size:
```bash
uv run jax-impl/node.py --max-batch-size 16
```

## Next Steps

- **Scheduler Integration**: Connect Rust scheduler to route requests to nodes
- **Continuous Batching**: Implement dynamic batch management
- **Prefix Caching**: Add cache state tracking and reporting to scheduler
- **Health Checks**: Implement periodic health monitoring
- **Metrics**: Add Prometheus metrics for monitoring

See `INFERENCE_SERVER.md` for full architecture documentation.
