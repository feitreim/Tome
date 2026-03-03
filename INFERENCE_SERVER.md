# Inference Server

A distributed LLM inference system with a Rust scheduler and MLX inference nodes. This architecture separates orchestration (Rust) from compute (Python/MLX) for optimal performance, specifically optimized for **GRPO (Group Relative Policy Optimization)** training loops.

## Architecture Overview

The system consists of:

- **Scheduler** (Rust): HTTP API server, prefix-aware routing, KV cache tracking, and weight update broadcasting.
- **Inference Nodes** (Python/MLX): Execute prefill, decode, and GRPO rollout operations. Uses a **Paged KV Cache** for efficient memory management.
- **Dual Model Setup**: Each node maintains two model copies (Policy and Reference) sharing the same physical memory pool for KV blocks.
- **Communication**: gRPC for scheduler-node communication, REST for client-scheduler communication.

## Scheduler

The scheduler receives HTTP requests and dispatches them to inference nodes based on prefix cache state and current load.

**Location**: `./scheduler/`

### Core Responsibilities

1. **REST API**: endpoints for completions, GRPO rollouts, and model weight updates.
2. **Prefix-Aware Routing**: Routes requests to nodes with matching cached prefixes (e.g., shared rubrics or prompts).
3. **Weight Management**: Broadcasts LoRA weight updates to all nodes for in-place merging.
4. **Health Monitoring**: Periodic gRPC pings to track node status and capacity.

## MLX Inference Nodes

Optimized for Apple Silicon unified memory, providing high-throughput batch inference.

**Location**: `./mlx-impl/`

### Paged KV Cache

Instead of contiguous buffers, KV pairs are stored in fixed-size blocks (default: 128 tokens).
- **BlockAllocator**: Manages a global pool of blocks across all active requests.
- **Copy-on-Write (CoW)**: Enables safe sharing of KV blocks between sequences. When a shared block is modified (e.g., during different rollouts from the same prompt), it is automatically cloned.
- **Radix Tree Prefix Cache**: Automatically caches and reuses blocks for common prefixes (rubrics, prompts).

### GRPO Training Support

The server provides a specialized `GRPO` RPC that handles the entire rollout-judge phase in a single optimized pipeline:

1. **Rollout Generation**: Generates $G$ completions for a batch of prompts using the Policy model.
2. **Judge Scoring**: Prefills a shared rubric and scores each rollout. Uses the 3-level prefix hierarchy (Rubric → Prompt → Completion) for maximum KV reuse.
3. **Reference Log-probs**: Computes $\log P_{ref}(\text{token} \mid \text{context})$ for all generated tokens using the frozen Reference model.

### In-place Weight Updates

Supports the `UpdateWeights` RPC to merge LoRA adapters ($B \times A$) directly into the active Policy model weights. This allows continuous training without restarting nodes or re-loading base weights.

## API Specification

### 1. Completions (`POST /v1/completions`)
Standard OpenAI-compatible completion endpoint.

### 2. GRPO Rollout (`POST /v1/grpo/rollout`)
Generates multiple rollouts per prompt and computes both policy and reference model log-probs.
- **Input**: Prompts, group size ($G$), and sampling parameters.
- **Output**: completions with per-token log-probs and reference log-probs.

### 3. GRPO Judge (`POST /v1/grpo/judge`)
Scores completions using a shared rubric and 3-level prefix caching.
- **Input**: Shared rubric, prompts, and completions to score.
- **Output**: Judge verdicts with per-token log-probs.

### 4. Weight Update (`POST /v1/weights`)
Updates the policy model weights on all nodes.
- **Input**: List of layer indices, parameter names, and base64-encoded LoRA matrices ($A$ and $B$).

## Implementation Status

- [x] Paged KV Cache with CoW
- [x] Radix Tree Prefix Caching
- [x] Dual Model Architecture (Policy + Reference)
- [x] In-place LoRA Weight Merging
- [x] Unified GRPO Pipeline (Rollout + Judge + Ref Logprobs)
- [x] Rust Scheduler Integration
