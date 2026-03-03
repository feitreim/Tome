# Scheduler

The scheduler is the central routing component of the Tome inference system. It accepts HTTP requests, selects the best inference node based on prefix cache state and load, and streams token responses back to the client via SSE.

## Building

```bash
cd scheduler
cargo build --release
```

This compiles `proto/inference.proto` automatically via `tonic-build` during the build step.

## Running

```bash
cargo run --release
```

The scheduler listens on `0.0.0.0:8080` by default.

### Configuration

| Environment Variable | Default          | Description                                                     |
| -------------------- | ---------------- | --------------------------------------------------------------- |
| `SCHEDULER_PORT`     | `8080`           | HTTP listen port                                                |
| `RUST_LOG`           | `scheduler=info` | Log level filter (uses `tracing-subscriber` `EnvFilter` syntax) |

Example with custom port and debug logging:

```bash
SCHEDULER_PORT=9000 RUST_LOG=scheduler=debug cargo run --release
```

## API

### `POST /v1/completions`

Send a completion request. Returns an SSE stream of token chunks (OpenAI-compatible format).

**Request:**

```json
{
  "prompt": "Once upon a time",
  "max_tokens": 100,
  "temperature": 0.7
}
```

| Field         | Type   | Default    | Description                |
| ------------- | ------ | ---------- | -------------------------- |
| `prompt`      | string | (required) | Input text                 |
| `max_tokens`  | u32    | `128`      | Maximum tokens to generate |
| `temperature` | f32    | `0.7`      | Sampling temperature       |

**Response** (SSE stream):

```
data: {"id":"<request-uuid>","choices":[{"text":"<token:1234>","finish_reason":null}]}

data: {"id":"<request-uuid>","choices":[{"text":"<token:5678>","finish_reason":"stop"}]}
```

Each SSE event contains a JSON chunk with a `choices` array. `finish_reason` is `null` while generating and `"stop"` on the final token.

If no healthy nodes are available, the stream returns a single error event:

```
data: {"error":"no healthy inference nodes available"}
```

**Example:**

```bash
curl -N -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_tokens": 50}'
```

### `POST /v1/nodes`

Register an inference node. The scheduler connects to the node's gRPC server and begins routing requests to it.

**Request:**

```json
{
  "addr": "http://10.0.0.2:50051"
}
```

**Response:**

```json
{
  "node_id": "http://10.0.0.2:50051",
  "status": "registered"
}
```

If the connection fails:

```json
{
  "error": "failed to connect to http://10.0.0.2:50051: ..."
}
```

**Example:**

```bash
curl -X POST http://localhost:8080/v1/nodes \
  -H "Content-Type: application/json" \
  -d '{"addr": "http://localhost:50051"}'
```

### `GET /v1/nodes`

List all registered nodes and their current status.

**Response:**

```json
{
  "nodes": [
    {
      "node_id": "http://localhost:50051",
      "addr": "http://localhost:50051",
      "healthy": true,
      "active_sequences": 3,
      "gpu_utilization": 0.65,
      "queue_depth": 1,
      "cached_tokens": 2048
    }
  ]
}
```

**Example:**

```bash
curl http://localhost:8080/v1/nodes
```

## Request Routing

The scheduler scores each healthy node using a weighted formula:

```
score = 10.0 * prefix_match_len + 1.0 * (1 - gpu_utilization) + 0.5 * (1 / queue_depth)
```

- **Prefix match** is heavily weighted -- reusing cached KV avoids expensive prefill compute.
- **GPU utilization** and **queue depth** act as tiebreakers to spread load.

If no node has a matching prefix, the least-loaded healthy node is selected.

## Health Checks

The scheduler pings every registered node via gRPC `GetStatus` every 5 seconds. Nodes that fail the health check are marked unhealthy and excluded from routing. They are automatically re-included once they start responding again.

## Inference Nodes

Nodes must implement the gRPC `InferenceNode` service defined in `proto/inference.proto`:

- `Prefill` -- process an input prompt, populate KV cache
- `Decode` -- generate one token from cached KV
- `StreamGenerate` -- full generation loop, streaming tokens back
- `GetStatus` -- report current load and cache state

See `mlx-impl/node.py` for the reference implementation.

## Shutdown

The scheduler shuts down on `SIGINT` (Ctrl+C) or `SIGTERM`. Active SSE streams will be terminated. In-flight gRPC requests to inference nodes are dropped. There is no graceful drain yet -- this is planned for Phase 4 (production hardening).
