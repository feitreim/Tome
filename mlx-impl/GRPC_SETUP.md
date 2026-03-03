# gRPC Inference Node Setup (MLX)

This guide explains how to set up and run the MLX inference node with gRPC server for Qwen3-0.6B on Apple Silicon.

## Prerequisites

1. **Install dependencies**:

   ```bash
   uv sync
   ```

   Note: MLX is included in the project dependencies and works exclusively on Apple Silicon (M1/M2/M3/M4).

2. **Generate gRPC Python code from protobuf**:

   ```bash
   uv run mlx-impl/generate_proto.py
   ```

   This creates Python files in `mlx-impl/generated/`:
   - `inference_pb2.py` - Protocol buffer message classes
   - `inference_pb2_grpc.py` - gRPC service stubs
   - `inference_pb2.pyi` - Type stubs for better IDE support

## Running the Server

### Basic Usage

Start the MLX inference node on default port 50052:

```bash
uv run mlx-impl/node.py
```

The default port is **50052**.

### With Custom Options

```bash
# Custom port
uv run mlx-impl/node.py --port 50053

# Custom checkpoint
uv run mlx-impl/node.py --checkpoint Qwen/Qwen3-0.6B

# All options
uv run mlx-impl/node.py \
  --port 50052 \
  --checkpoint Qwen/Qwen3-0.6B \
  --max-batch-size 32
```

### Command-line Options

- `--port`: gRPC server port (default: 50052)
- `--checkpoint`: HuggingFace model checkpoint (default: Qwen/Qwen3-0.6B)
- `--max-batch-size`: Maximum batch size for continuous batching (default: 32)

## Testing the Server

### Using the Test Client

In a separate terminal, run the test client:

```bash
uv run mlx-impl/test_grpc_client.py
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
```

List available services:

```bash
grpcurl -plaintext localhost:50052 list
```

Get node status:

```bash
grpcurl -plaintext localhost:50052 inference.InferenceNode/GetStatus
```

Send prefill request:

```bash
grpcurl -plaintext -d '{
  "request_id": "test-mlx-001",
  "tokens": [151643, 9906, 11, 856, 836, 374],
  "temperature": 0.7,
  "max_tokens": 100
}' localhost:50052 inference.InferenceNode/Prefill
```

## Architecture

### Protocol Buffers Schema

The gRPC interface is defined in `scheduler/proto/inference.proto`.

### Implementation Details

**File**: `mlx-impl/node.py`

- `InferenceNodeServicer` - Implements gRPC service
  - Manages KV cache using `kvcache.py`
  - Tracks active sequences for continuous batching
  - Supports temperature-based sampling with MLX random ops

- `serve()` - Async server entry point
  - Loads Qwen3-0.6B model from HuggingFace
  - Starts gRPC server on specified port

### KV Cache Implementation

MLX implementation in `./kvcache.py`:

```python
class KVCache:
    def __init__(self, num_layers, num_kv_heads, head_dim, max_seq_len):
        self.keys = [None] * num_layers
        self.values = [None] * num_layers
        self.offset = 0

    def update(self, k, v, layer_idx):
        # Concatenate new keys/values to cache
        # Returns full cached keys and values
```

## Performance Characteristics

### MLX Framework Benefits

MLX is optimized for Apple Silicon:

- **Unified memory**: No CPU↔GPU transfers needed
- **Metal GPU acceleration**: Native Metal kernels
- **Fast kernels**: `mx.fast.rope()`, `mx.fast.scaled_dot_product_attention()`, `mx.fast.rms_norm()`
- **Lazy evaluation**: Computations happen on `mx.eval()`

### Benchmarks

See `MLX.md` for detailed benchmarks. Summary:

- **Prefill**: ~1,100 tokens/s (512-token prompts)
- **Decode**: ~36 tokens/s (single batch, memory-bound)
- **Memory**: ~2-3 GB total (fits comfortably on 8GB M1)

## Troubleshooting

### Proto files not found

If you get import errors for `inference_pb2`, make sure you generated the proto files:

```bash
uv run mlx-impl/generate_proto.py
```

### Port already in use

Change the port:

```bash
uv run mlx-impl/node.py --port 50053
```

### Model download fails

Ensure you have internet connection. First run downloads from HuggingFace (~1.2GB model):

```bash
# Model will be cached in ~/.cache/huggingface/
ls -lh ~/.cache/huggingface/hub/
```

## Next Steps

- **Scheduler Integration**: Connect Rust scheduler to route requests to nodes
- **Continuous Batching**: Implement dynamic batch management
- **Cache Persistence**: Store KV cache between requests (currently recreated)
- **Health Checks**: Implement periodic health monitoring
- **Metrics**: Add Prometheus metrics for monitoring
- **Metal Monitoring**: Add GPU utilization tracking for Apple Silicon

See `../INFERENCE_SERVER.md` for full architecture documentation.
