# gRPC Inference Node

This document provides an overview of the MLX gRPC inference node for distributed LLM serving.

## MLX Implementation (Apple Silicon)

**Location**: `mlx-impl/`

**Model**: Nanbeige4.1-3B (Dense Transformer)

- 3B parameters
- 32 decoder layers, Grouped Query Attention
- ~6 GB memory footprint (bfloat16 weights)
- Apple Silicon only (M1/M2/M3/M4)

**Default Port**: 50052

**Key Features**:

- Optimized Metal GPU kernels
- Custom fused norm+RoPE kernel
- Grouped Query Attention (20 Q heads, 4 KV heads)
- Unified memory — no CPU/GPU transfers

**Quick Start**:

```bash
# Generate proto files
uv run mlx-impl/generate_proto.py

# Start server
./start_mlx_node.sh

# Or directly
uv run mlx-impl/node.py
```

## gRPC Service API

The node implements the `InferenceNode` gRPC service defined in `scheduler/proto/inference.proto`:

| RPC                | Description                                                |
| ------------------ | ---------------------------------------------------------- |
| **Prefill**        | Process input prompt, populate KV cache, return next token |
| **Decode**         | Generate one token from cached KV state                    |
| **StreamGenerate** | Full generation loop, streaming tokens back                |
| **GetStatus**      | Report current load and cache state                        |

## File Structure

```
.
├── scheduler/
│   └── proto/
│       └── inference.proto        # gRPC schema
│
├── mlx-impl/                      # MLX implementation
│   ├── node.py                    # gRPC server (Nanbeige4.1-3B)
│   ├── model.py                   # Model + Metal kernels
│   ├── load_weights.py            # SafeTensors weight loading
│   ├── kvcache.py                 # KV cache
│   ├── generate_proto.py          # Proto code generator
│   ├── test_grpc_client.py        # Test client
│   ├── generated/                 # Auto-generated gRPC code
│   ├── README.md                  # Quick reference
│   └── GRPC_SETUP.md             # Detailed setup guide
│
├── start_mlx_node.sh             # MLX node launcher
└── GRPC_NODES.md                 # This file
```

## Testing

```bash
# Start server
uv run mlx-impl/node.py

# In another terminal
uv run mlx-impl/test_grpc_client.py
```

### Using grpcurl

```bash
grpcurl -plaintext localhost:50052 list
grpcurl -plaintext localhost:50052 inference.InferenceNode/GetStatus
```

## Development Workflow

### 1. Modify Protocol Buffers

Edit `scheduler/proto/inference.proto`

### 2. Regenerate Code

```bash
uv run mlx-impl/generate_proto.py
```

### 3. Update Implementation

Modify `mlx-impl/node.py`

### 4. Lint and Format

```bash
uvx ruff check mlx-impl/
uvx ruff format mlx-impl/
```

## Architecture Integration

The gRPC node integrates with the Rust scheduler (see `INFERENCE_SERVER.md`):

```
┌─────────────────────┐
│   HTTP Clients      │
│  (OpenAI API)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Rust Scheduler     │
│  (HTTP → gRPC)      │
│  - Prefix caching   │
│  - Load balancing   │
│  - Trie tracking    │
└──────────┬──────────┘
           │
           ├──────────────────────┐
           │                      │
           ▼                      ▼
    ┌──────────┐          ┌──────────┐
    │MLX Node 1│          │MLX Node 2│
    │Port 50052│          │Port 50053│
    │M3 Max    │          │M4 Pro    │
    └──────────┘          └──────────┘
```

The scheduler routes requests to nodes based on:

- Prefix cache hits
- Current load
- Hardware availability

## Next Steps

1. **Continuous Batching** — dynamic batch management in nodes
2. **Cache Persistence** — store KV cache between requests
3. **Health Checks** — periodic heartbeat monitoring
4. **Metrics** — Prometheus metrics for monitoring
5. **Cache State Reporting** — report cached prefixes to scheduler for prefix-aware routing

See `INFERENCE_SERVER.md` for the detailed architecture and roadmap.

## Resources

### Documentation

- `INFERENCE_SERVER.md` — Overall architecture
- `mlx-impl/README.md` — MLX node quick reference
- `mlx-impl/GRPC_SETUP.md` — Setup guide
- `mlx-impl/MLX.md` — Model implementation details and benchmarks

### Framework Documentation

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [gRPC Python](https://grpc.io/docs/languages/python/)
