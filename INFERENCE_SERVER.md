# Inference Server

A distributed LLM inference system with a Rust scheduler and MLX inference nodes. This architecture separates orchestration (Rust) from compute (Python/MLX) for optimal performance.

## Architecture Overview

The system consists of:

- **Scheduler** (Rust): HTTP API server, prefix-aware routing, KV cache tracking
- **Inference Nodes** (Python/MLX): Execute prefill and decode operations, maintain local KV cache
- **Communication**: gRPC for scheduler-node communication

## Scheduler

The scheduler receives HTTP requests (REST API) and dispatches them to inference nodes. It is multithreaded with async I/O and implements intelligent load balancing based on KV cache state.

**Location**: `./scheduler/`

### Core Responsibilities

1. **HTTP API Server**: Accept inference requests (OpenAI-compatible API format)
2. **Prefix-Aware Routing**: Route requests to nodes with matching cached prefixes
3. **KV Cache Tracking**: Maintain a trie structure tracking what tokens are cached on each node
4. **Load Balancing**: Balance between cache hit rate and node utilization
5. **Request Queuing**: Queue requests when all nodes are busy

### Communication Protocol: gRPC

**Recommendation**: Use gRPC for scheduler-to-node communication.

**Rationale**:

- Built-in service discovery, health checking, and load balancing
- Strongly-typed interfaces via Protocol Buffers (prevents serialization bugs)
- Bidirectional streaming for efficient token streaming
- Better suited for service-oriented architecture than raw TCP or ZeroMQ
- Industry standard for distributed systems (used by vLLM, Ray Serve)

**Alternative Considered**: ZeroMQ offers lower latency but lacks built-in service discovery and typed interfaces, requiring more manual infrastructure.

**Dependencies to add**:

```toml
# Cargo.toml
[dependencies]
tonic = "0.12"           # gRPC framework
prost = "0.13"           # Protocol Buffers
tokio = { version = "1", features = ["full"] }
tower = "0.5"            # Service abstractions
```

### Prefix Cache Tracking with Trie

The scheduler tracks the KV cache contents of each node using a **radix trie** (prefix tree), similar to SGLang's approach. This enables efficient prefix matching for cache reuse.

**Data Structure**:

```rust
struct TrieNode {
    token_id: Option<u32>,
    children: HashMap<u32, TrieNode>,
    cached_nodes: HashSet<NodeId>,  // Which nodes have this prefix cached
}

struct CacheTracker {
    root: TrieNode,
    node_states: HashMap<NodeId, NodeState>,
}
```

**Operations**:

- `insert_prefix(node_id, tokens)`: Track that a node has cached a token sequence
- `find_best_match(tokens) -> (NodeId, prefix_len)`: Find node with longest matching prefix
- `evict_prefix(node_id, tokens)`: Remove cached prefix when node evicts from KV cache

**Routing Algorithm**:

1. Query trie to find nodes with matching prefixes
2. Select node with longest prefix match
3. If tie, use node with lowest current load
4. If no prefix match, use least-loaded node

This approach can achieve **15× higher throughput** and **2× lower latency** compared to round-robin routing (see LMCACHE research).

### Load Balancing Strategy

**Multi-Factor Scoring**:

```
score(node, request) = w1 * prefix_match_len
                     + w2 * (1 - current_load)
                     + w3 * (1 / queue_depth)
```

Where:

- `prefix_match_len`: Number of cached tokens matching request prefix
- `current_load`: GPU utilization (0.0 to 1.0)
- `queue_depth`: Number of queued requests for this node

**Weights** (suggested starting values):

- `w1 = 10.0`: Heavily favor cache hits (saves compute)
- `w2 = 1.0`: Moderately favor less-loaded nodes
- `w3 = 0.5`: Slightly penalize deep queues

This is similar to the llm-d project's "prefix-aware intelligent routing" which reduces P95/P99 latency by up to 40%.

## KV Cache Management

### Request Types

**Prefill**: Process input prompt, populate KV cache, return logits for next token

- Compute-bound (large matrix multiplications)
- High latency (processes all tokens in parallel)
- Returns: next token logits + KV cache state

**Decode**: Generate one token autoregressively using cached KV

- Memory-bound (attention over cached keys/values)
- Low latency (processes single token)
- Returns: next token logits

### Continuous Batching

Nodes should implement **continuous batching** (iteration-level scheduling):

- Don't wait for entire batch to finish
- Replace completed sequences immediately with new requests
- Achieves **23× higher throughput** vs static batching (vLLM benchmarks)

**Implementation Strategy**:

- Node maintains active batch of sequences
- After each decode step, remove finished sequences
- Scheduler sends new requests to fill empty slots
- Requires careful attention masking to handle variable sequence lengths

### Prefill/Decode Disaggregation (Future Work)

**Optional Advanced Feature**: Run separate node pools for prefill and decode.

**Benefits**:

- Optimize hardware separately (prefill needs compute, decode needs memory bandwidth)
- Up to **6.4× throughput** and **20× lower latency variance** (DistServe research)
- Used in production by Meta, LinkedIn, Mistral

**Implementation**:

1. Prefill nodes: Process prompts, generate KV cache
2. Transfer KV cache to decode nodes via gRPC streaming
3. Decode nodes: Generate tokens using transferred cache

**Challenge**: Fast KV cache transfer between nodes (requires high-bandwidth network).

## Inference Nodes (Python/MLX)

**Location**: `./mlx-impl/node.py`

### Responsibilities

1. **gRPC Server**: Listen for requests from scheduler
2. **Model Execution**: Run prefill or decode on OLMoE model
3. **KV Cache Management**: Maintain local cache, handle eviction
4. **Continuous Batching**: Manage active sequence batch
5. **Token Streaming**: Stream generated tokens back to scheduler

### Communication: gRPC Client/Server

**Python Dependencies**:

```bash
pip install grpcio grpcio-tools
```

**Protocol Buffer Schema** (`inference.proto`):

```protobuf
service InferenceNode {
  rpc Prefill(PrefillRequest) returns (PrefillResponse);
  rpc Decode(DecodeRequest) returns (DecodeResponse);
  rpc StreamGenerate(GenerateRequest) returns (stream TokenResponse);
  rpc GetStatus(StatusRequest) returns (NodeStatus);
}

message PrefillRequest {
  string request_id = 1;
  repeated uint32 tokens = 2;
  float temperature = 3;
  uint32 max_tokens = 4;
}

message DecodeRequest {
  string request_id = 1;
  uint32 cache_position = 2;
}

message TokenResponse {
  string request_id = 1;
  uint32 token_id = 2;
  bool is_finished = 3;
}

message NodeStatus {
  uint32 active_sequences = 1;
  float gpu_utilization = 2;
  uint32 queue_depth = 3;
  uint64 cached_tokens = 4;
}
```

### KV Cache Implementation

MLX implementation is in `./mlx-impl/kvcache.py`. The node needs to:

1. **Allocate cache** on startup: `KVCache.new(layers=16, max_seq_len=2048, ...)`
2. **Update cache** during prefill/decode: `cache.update(k, v, layer_num, cur_pos, n_reps)`
3. **Evict old sequences** when cache is full (FIFO or LRU policy)
4. **Report cache state** to scheduler (which prefixes are cached)

### Batching Strategy

Implement continuous batching with dynamic batch size:

```python
class InferenceNode:
    def __init__(self, max_batch_size: int = 32):
        self.active_batch: list[Sequence] = []
        self.max_batch_size = max_batch_size

    async def process_step(self):
        """Single decode step for all active sequences"""
        # Generate next tokens for all sequences in batch
        # Remove finished sequences
        # Fill empty slots from queue
```

## Request/Response Flow

### Example: Generate Request

1. **Client → Scheduler** (HTTP):

   ```json
   POST /v1/completions
   {
     "prompt": "Once upon a time",
     "max_tokens": 100,
     "temperature": 0.7
   }
   ```

2. **Scheduler** (internal):
   - Tokenize prompt: `[1234, 5678, 9012, 3456]`
   - Query trie: Find node with best prefix match
   - Route to selected node

3. **Scheduler → Node** (gRPC):

   ```protobuf
   PrefillRequest {
     request_id: "req-123"
     tokens: [1234, 5678, 9012, 3456]
     temperature: 0.7
     max_tokens: 100
   }
   ```

4. **Node** (internal):
   - Run prefill: populate KV cache for prompt
   - Sample next token
   - Add to active batch for continuous decoding

5. **Node → Scheduler** (gRPC streaming):

   ```protobuf
   TokenResponse { request_id: "req-123", token_id: 7890, is_finished: false }
   TokenResponse { request_id: "req-123", token_id: 2345, is_finished: false }
   ...
   TokenResponse { request_id: "req-123", token_id: 6789, is_finished: true }
   ```

6. **Scheduler → Client** (HTTP streaming):
   ```
   data: {"id": "req-123", "choices": [{"text": " there", "finish_reason": null}]}
   data: {"id": "req-123", "choices": [{"text": " was", "finish_reason": null}]}
   ...
   data: {"id": "req-123", "choices": [{"text": ".", "finish_reason": "length"}]}
   ```

## Error Handling & Fault Tolerance

### Node Failures

**Detection**: Scheduler pings nodes with health checks (gRPC `GetStatus` RPC every 5s)

**Recovery**:

- Mark node as unhealthy, stop routing new requests
- Re-queue in-flight requests to other nodes (prefix cache may not match, but ensures progress)
- If node recovers, gradually reintroduce to rotation

### Request Timeouts

- Client timeout: Return 504 Gateway Timeout after configured deadline
- Node timeout: If node doesn't respond within SLA, cancel request and try different node
- Track timeout rates per node, deprioritize unreliable nodes

### KV Cache Invalidation

If node crashes or evicts cache:

- Node reports cache state changes to scheduler
- Scheduler updates trie to remove invalidated prefixes
- Future requests won't be routed based on stale cache info

## Performance Optimizations

### 1. Speculative Prefill Reuse

Cache popular prompt prefixes (e.g., system prompts, common instructions) across all nodes for instant cache hits.

### 2. Request Coalescing

If multiple requests share exact same prefix, process them together in a single batch.

### 3. Chunked Prefill

For very long prompts, break prefill into chunks to reduce latency before first token (start decode earlier).

### 4. GPU Memory Management

Use MLX's unified memory architecture and lazy evaluation to minimize allocation overhead during continuous batching.

## References & Further Reading

**Communication Protocols**:

- [ZeroMQ vs gRPC comparison](https://stackshare.io/stackups/grpc-vs-zeromq)
- [Comparative Analysis of gRPC vs ZeroMQ](https://www.researchgate.net/publication/389078536_Comparative_Analysis_OF_GRPC_VS_ZeroMQ_for_Fast_Communication)

**Prefix Caching**:

- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [LMCACHE: Efficient KV Cache for Enterprise-Scale LLM Inference](https://lmcache.ai/tech_report.pdf)
- [LLM Inference Handbook: Prefix Caching](https://bentoml.com/llm/inference-optimization/prefix-caching)

**Continuous Batching**:

- [vLLM: High-throughput and memory-efficient inference](https://github.com/vllm-project/vllm)
- [Continuous Batching for 23x LLM Inference Throughput](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [Inside vLLM: Anatomy of High-Throughput LLM System](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)

**Intelligent Routing**:

- [llm-d: Intelligent routing with KV-cache awareness](https://github.com/llm-d/llm-d)
- [Accelerate multi-turn workloads with llm-d](https://developers.redhat.com/articles/2026/01/13/accelerate-multi-turn-workloads-llm-d)
- [How to Create LLM Load Balancing](https://oneuptime.com/blog/post/2026-01-30-llmops-load-balancing/view)

**Prefill/Decode Disaggregation**:

- [vLLM Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- [DistServe: Disaggregating Prefill and Decoding](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)
- [BentoML: Prefill-decode disaggregation](https://bentoml.com/llm/inference-optimization/prefill-decode-disaggregation)
- [Meta's LLM Serving with Disaggregated Prefill-Decode](https://docs.jarvislabs.ai/blog/llm-optimization-disaggregated-prefill-decode)

## Implementation Roadmap

### Phase 1: Basic Infrastructure

- [ ] Scheduler: HTTP server accepting requests
- [ ] Node: gRPC server with prefill/decode RPCs
- [ ] Protocol Buffers schema definition
- [ ] Basic round-robin load balancing

### Phase 2: Prefix Caching

- [ ] Scheduler: Trie-based cache tracking
- [ ] Node: Report cached prefixes to scheduler
- [ ] Prefix-aware routing algorithm
- [ ] Cache eviction policies

### Phase 3: Continuous Batching

- [ ] Node: Dynamic batch management
- [ ] Iteration-level scheduling
- [ ] Token streaming via gRPC

### Phase 4: Production Hardening

- [ ] Health checks and fault tolerance
- [ ] Metrics and monitoring (Prometheus)
- [ ] Request timeout handling
- [ ] Load testing and optimization

### Phase 5: Advanced Features (Optional)

- [ ] Prefill/decode disaggregation
- [ ] Speculative decoding
- [ ] Multi-node tensor parallelism
