# Tome Feature Plan

Three features for mlx-impl. Target model: **Qwen 0.6B** (bf16, ~1.2 GB weights — fits comfortably in Apple Silicon unified memory, leaves room for KV). Focus: efficient GRPO training loop with prefix-shared rollouts and judge scoring.

---

## 1. Paged KV Cache

Efficient memory management for high concurrency. Replace the current contiguous-allocation KVCache.

### Current State

- `kvcache.py`: Per-sequence contiguous buffers, grows by 128 slots when full
- Each sequence gets its own `KVCache` instance with `list[mx.array | None]` per layer
- No memory sharing, no eviction policy, no upper bound on total memory

### Design

**Block table**: Allocate KV memory in fixed-size blocks (e.g. 64 or 128 tokens). Maintain a block table per sequence mapping logical positions to physical block indices. On decode, only allocate a new block when the current one fills up.

```
Sequence "req-42":
  logical blocks: [0, 1, 2, 3]
  physical blocks: [17, 3, 91, 44]
```

**Free list + allocator**: Global pool of physical blocks. Allocate from free list, return on sequence completion or eviction. Track total memory budget and reject/queue new requests when full.

**Copy-on-write**: If two sequences share a prefix, they share physical blocks for the common portion. Reference count blocks, allocate new blocks only when they diverge. Critical for both GRPO rollouts (G sequences share prompt KV) and judge scoring (all scores for one prompt share rubric+prompt KV).

**Eviction policies**:
- LRU: evict least recently used sequence's blocks when pool is exhausted
- Priority-based: protect sequences close to completion, evict sequences that just started (cheaper to re-prefill)

**MLX considerations**: MLX uses unified memory and lazy evaluation. The block table approach needs `mx.take` or scatter/gather to assemble non-contiguous blocks into contiguous KV for SDPA. Benchmark whether the gather overhead is worth the memory savings, or whether a simpler approach (pre-allocated large contiguous buffer with slot assignment) is better for MLX's memory model.

---

## 2. Prefix Caching

Radix tree-based caching for shared prefixes — within a GRPO batch (rollout groups, judge groups) and across batches (rubric reuse).

### Current State

- Scheduler has a trie design in `INFERENCE_SERVER.md` for tracking which node has which prefix
- `CACHE.md` notes: pack token IDs as bytes for fast parallel prefix comparison
- No actual prefix cache implementation yet — each request does a full prefill

### Design

**Radix tree on the node**: Store completed KV cache blocks in a radix tree keyed by token sequence. On new request, walk the tree to find the longest matching prefix, reuse those KV blocks (CoW), only prefill the suffix.

**Three-level hierarchy for GRPO judge scoring**:

```
Root
└── [rubric...] → blocks [0..3]                         # shared across ALL judge calls
    ├── [prompt_A...] → blocks [4..6]                    # shared across 16 completions for A
    │   ├── [completion_A_1...] → blocks [7..10]         # unique suffix
    │   ├── [completion_A_2...] → blocks [7, 11..13]     # diverges partway
    │   └── ...
    ├── [prompt_B...] → blocks [14..16]
    │   └── ...
    └── ...
```

The rubric prefix is prefilled once for the entire batch. Each prompt extends it, prefilled once per prompt. Each completion extends that, prefilled once per completion. Maximum KV reuse.

**Eviction integration with paged cache**: Prefix cache entries are pinned blocks in the paged KV pool. Under memory pressure, unpin and evict cold prefixes (LRU by last access time). The rubric prefix stays hot across the entire training run.

**Parallel prefix check** (from CACHE.md): Pack token IDs into byte streams. Compare incoming prompt against all cached byte streams in parallel — the one that matches longest wins. Can be SIMD-accelerated.

---

## 3. GRPO Training Loop

Full GRPO pipeline: chunked rollout generation, judge scoring, advantage computation. All prefix-shared via paged KV + radix tree.

### Pipeline

A GRPO request: **P prompts × G rollouts each** (e.g. 8 prompts × 16 rollouts = 128 total completions). Memory-bounded: only N×G fit at once (e.g. 4×16 = 64 sequences). Process in chunks, maximize prefix sharing within each chunk.

#### Phase 1: Rollout Generation (chunked)

```
Request: 8 prompts × 16 rollouts
Memory limit: 4 prompts × 16 rollouts = 64 active sequences

Chunk 1: prompts [A, B, C, D] × 16 rollouts
  - Prefill A once → fork 16 sequences (CoW)
  - Prefill B once → fork 16 sequences (CoW)
  - Prefill C once → fork 16 sequences (CoW)
  - Prefill D once → fork 16 sequences (CoW)
  - Decode all 64 in parallel until EOS/max_tokens
  - Collect tokens + log_probs per completion
  - Free KV blocks for A/B/C/D rollouts

Chunk 2: prompts [E, F, G, H] × 16 rollouts
  - Same pattern
```

Each chunk: 4 prefills instead of 64 (16x reduction). All 16 rollouts per prompt share the prompt's KV blocks via CoW, only diverging once they start generating different tokens.

**Log-prob collection**: During decode, collect `log P(token | context)` for every generated token. Use the existing `fused_log_softmax` Metal kernel on the logits, then index by the sampled token. Store per-sequence alongside token IDs.

**Early termination**: When a sequence hits EOS, stop decoding it but keep its slot until the chunk completes (simpler than backfilling, and all 16 rollouts for a prompt tend to be similar length). Pad log_probs with zeros for terminated sequences.

#### Phase 2: Judge Scoring (chunked)

Score each completion. The judge prompt is: `rubric + original_prompt + completion`. Same chunking as rollouts.

```
Chunk 1: judge calls for prompts [A, B, C, D] × 16 completions
  - Prefill rubric once → shared across all 64 (CoW)
  - Prefill rubric+prompt_A once → shared across 16 (CoW)
  - Prefill rubric+prompt_A+completion_A_1 suffix → unique
  - ... (15 more for prompt A)
  - Same for B, C, D
  - Decode all 64 to get judge verdicts

Chunk 2: prompts [E, F, G, H]
  - Rubric prefix is still cached from chunk 1 (radix tree hit)
  - Only need to prefill prompt + completion suffixes
```

Cross-chunk prefix reuse: the rubric KV stays in the radix tree between chunks. Second chunk gets it for free.

#### Phase 3: Advantage Computation

After all chunks complete (all 128 rollouts generated and scored):

```python
# Per-group (per-prompt) advantage normalization
for prompt in prompts:
    rewards = scores[prompt]                              # shape: (G,)
    advantages = (rewards - rewards.mean()) / (rewards.std() + eps)

# GRPO loss (computed by training loop, not the inference node)
# L = -E[ advantage * (log_prob - log_prob_ref) ] + beta * KL(policy || ref)
```

### API

```protobuf
service InferenceNode {
  // ...
  rpc Rollout(RolloutRequest) returns (RolloutResponse);
  rpc Judge(JudgeRequest) returns (JudgeResponse);
}

message RolloutRequest {
  string batch_id = 1;
  repeated Prompt prompts = 2;
  uint32 group_size = 3;              // G rollouts per prompt
  float temperature = 4;
  uint32 max_tokens = 5;
}

message RolloutResponse {
  string batch_id = 1;
  repeated PromptRollout results = 2;
}

message PromptRollout {
  string prompt_id = 1;
  repeated Completion completions = 2;
}

message Completion {
  repeated uint32 tokens = 1;
  repeated float log_probs = 2;
  repeated float ref_log_probs = 3; // Policy rollouts through frozen model
}

message JudgeRequest {
  string batch_id = 1;
  repeated uint32 rubric_tokens = 2;  // Shared rubric prefix
  repeated JudgeItem items = 3;
  float temperature = 4;
  uint32 max_tokens = 5;
}

message JudgeItem {
  string item_id = 1;
  repeated uint32 prompt_tokens = 2;  // Suffix: Prompt + Completion
}

message JudgeResponse {
  string batch_id = 1;
  repeated JudgeResult results = 2;
}

message JudgeResult {
  string item_id = 1;
  repeated uint32 verdict_tokens = 2;
  repeated float log_probs = 3;
}
```

---

## 4. Weight Management (LoRA Merge + Reference Model)

Tome is inference-only. An external async trainer owns the LoRA training loop (computes A, B matrices). Tome receives weight updates and applies them at full rank — no adapter overhead in the forward pass.

### Design

**Two model copies in memory**:
- **Policy model** (~1.2 GB): actively updated, used for rollout generation
- **Reference model** (~1.2 GB): frozen at base weights, used for reference log-prob computation

Total: ~2.4 GB for weights, well within Apple Silicon memory budget.

**Weight update flow**:

```
Trainer                          Tome (inference node)
   │                                  │
   │  LoRA update (A, B matrices)     │
   │  or merged delta W_delta = BA    │
   │ ─────────────────────────────▶   │
   │                                  │  W_policy = W_base + W_delta
   │                                  │  (apply to policy model in-place)
   │                                  │  W_reference unchanged
   │                                  │
   │  Request: rollouts + ref logprobs│
   │ ─────────────────────────────▶   │
   │                                  │  1. Generate rollouts w/ policy model
   │                                  │  2. Compute ref log-probs w/ reference model
   │  completions + log_probs         │     (forward pass over generated tokens)
   │  + ref_log_probs + judge_scores  │
   │ ◀─────────────────────────────   │
```

**Update protocol**: The trainer sends weight updates via gRPC. Two options for what gets sent:

1. **Send A and B** — Tome computes `W_policy[layer] = W_reference[layer] + B @ A` per target layer. Smaller transfer (rank r << dim), merge is a single matmul per layer.
2. **Send merged delta** — Trainer pre-computes `BA`, sends the full delta. Larger transfer but no compute on Tome's side. Just `W_policy += delta`.

Option 1 is better for bandwidth (especially over network). The merge cost is negligible — for Qwen 0.6B with rank 16, it's a few small matmuls.

**Which layers get LoRA**: Typically `q_proj`, `k_proj`, `v_proj`, `o_proj`, and optionally `gate_proj`, `up_proj`, `down_proj`. The update message specifies layer index + parameter name + A/B tensors.

**Reference log-prob pass**: After rollout generation completes for a chunk, run a forward pass of the reference model over the generated token sequences to get `log P_ref(token | context)`. This reuses the same prefill+decode infrastructure — just with the frozen weights and no sampling (teacher-forcing the already-generated tokens). The prefix cache can help here too: if the prompt prefix was already cached during rollout, the reference model pass still needs its own KV (different weights = different KV values), but the compute pattern is identical.

**Atomicity**: Weight updates must not happen mid-generation. Apply updates between GRPO chunks — finish current chunk, apply update, start next chunk with new policy weights. Invalidate the policy model's prefix cache on update (KV values change when weights change). Reference model's prefix cache is never invalidated.

```protobuf
message WeightUpdateRequest {
  repeated LayerUpdate updates = 1;
}

message LayerUpdate {
  uint32 layer_idx = 1;
  string param_name = 2;           // e.g. "self_attn.q_proj"
  bytes lora_A = 3;                // (rank, in_dim) as bf16
  bytes lora_B = 4;                // (out_dim, rank) as bf16
}

message WeightUpdateResponse {
  bool success = 1;
  uint64 policy_version = 2;       // monotonic version counter
}
```

---

## Cross-Cutting: Model Switch

Switch from Nanbeige4.1-3B to **Qwen 0.6B** (Qwen/Qwen3-0.6B or similar). Same Qwen3 architecture class — just different config values. The `model.py` Qwen3 class already supports the relevant config flags. Need to:
- Add Qwen 0.6B config (dim, layers, heads, etc.)
- Update `load_weights.py` for the checkpoint
- Update tests/benchmarks for new model size
- Re-measure KL divergence

---

## Implementation Order

**1. Paged KV Cache** — block allocator, block table, CoW reference counting
**2. Prefix Caching** — radix tree over paged blocks, lookup/insert/evict
**3. Weight Management** — dual model setup, LoRA merge RPC, ref log-prob pass
**4. GRPO Rollouts** — chunked generation with prefix-shared CoW, log-prob collection
**5. GRPO Judge** — chunked scoring with three-level prefix hierarchy
**6. GRPO Orchestrator** — tie it together: chunk scheduling, advantage computation, API

---

## Benchmarking vs mlx-vllm

Compare Tome against [mlx-vllm](https://github.com/nils-org/mlx-vllm) across the workloads that matter for GRPO training.

### Metrics

- **Prefill throughput** (tok/s): single prompt, varying lengths (128, 512, 2048, 8192)
- **Decode throughput** (tok/s): single sequence, measure sustained generation speed
- **Batched decode throughput** (tok/s aggregate): 16, 32, 64 concurrent sequences
- **Prefix sharing speedup**: Time to generate G=16 rollouts for one prompt, with vs without shared prefill. mlx-vllm has automatic prefix caching — compare against our CoW paged approach
- **TTFT** (time to first token): prompt in → first generated token out
- **Peak memory**: RSS during batched generation at various concurrency levels
- **GRPO batch throughput**: End-to-end time for a full GRPO step (P prompts × G rollouts + judge scoring). This is the number that actually matters — mlx-vllm would need to do P×G independent requests vs our prefix-shared chunked approach

### Test Configurations

| Config | Prompts | Rollouts | Concurrent | Total Sequences |
|--------|---------|----------|------------|-----------------|
| Small  | 2       | 8        | 2×8 = 16   | 16              |
| Medium | 4       | 16       | 4×16 = 64  | 64              |
| Large  | 8       | 16       | 4×16 = 64  | 128 (2 chunks)  |

### What We Expect to Win On

- **Prefix-shared rollouts**: mlx-vllm's prefix caching works across requests but isn't optimized for the "one prompt, G forks" pattern. CoW blocks should give us near-zero overhead for the shared portion.
- **Judge scoring**: Three-level prefix hierarchy (rubric → prompt → completion) is GRPO-specific. mlx-vllm would redundantly prefill the rubric for every judge call.
- **Memory efficiency**: Paged KV with CoW means 16 rollouts from the same prompt use ~1x prompt KV + 16x divergent suffix KV, vs 16x full KV in a naive approach.
- **Weight update latency**: In-place LoRA merge between chunks with no model reload. mlx-vllm would need to restart or reload.

### What mlx-vllm Might Win On

- **Raw single-sequence decode speed**: mlx-vllm is mature and well-optimized. Our custom kernels may not beat their tuned Metal paths for single-stream decode.
- **Scheduler overhead**: Their continuous batching scheduler is battle-tested. Our chunked approach is simpler but may leave GPU idle between chunks.
- **Quantization support**: If mlx-vllm supports int4/int8, they can fit more concurrent sequences in the same memory budget.
