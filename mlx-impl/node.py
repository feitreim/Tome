"""MLX inference node with gRPC server for Nanbeige4.1-3B."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import sys
import time
from concurrent import futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib import request as urlrequest

import grpc
import mlx.core as mx
import numpy as np
from kvcache import BlockAllocator, KVCache, PrefixCache
from load_weights import download_qwen3, load_qwen3_weights
from model import Qwen3

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# Import generated gRPC code
sys.path.insert(0, str(Path(__file__).parent / "generated"))
import inference_pb2
import inference_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MAX_INFLIGHT_ROLLOUTS = 64


@dataclass
class ActiveSequence:
    """Represents an active generation sequence in the batch."""

    request_id: str
    tokens: list[int]
    temperature: float
    max_tokens: int
    cache: KVCache = field(default_factory=lambda: None)
    is_finished: bool = False


class InferenceNodeServicer(inference_pb2_grpc.InferenceNodeServicer):
    """gRPC servicer for MLX inference node."""

    def __init__(
        self,
        model: Qwen3,
        reference_model: Qwen3,
        max_inflight_rollouts: int = DEFAULT_MAX_INFLIGHT_ROLLOUTS,
        max_seq_len: int = 2048,
        vocab_size: int = 166144,
        eos_token_id: int = 166101,
        num_blocks: int = 1024,  # 1k blocks * 128 tokens = 128k tokens
        block_size: int = 128,
    ):
        self.model = model  # Policy model (active)
        self.reference_model = reference_model  # Reference model (frozen)
        self.max_inflight_rollouts = max_inflight_rollouts
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.policy_version = 0

        # Initialize KV cache parameters
        self.num_layers = model.num_layers
        self.num_kv_heads = model.num_kv_heads
        self.head_dim = model.head_dim

        # Global Block Allocator and Prefix Cache
        self.allocator = BlockAllocator(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )
        self.prefix_cache = PrefixCache(self.allocator)
        # Separate prefix cache for reference model?
        # Actually reference weights are frozen, so prefix cache is always valid.
        self.reference_prefix_cache = PrefixCache(self.allocator)

        # Active sequences being processed
        self.active_sequences: dict[str, ActiveSequence] = {}
        self.sequence_queue: asyncio.Queue[ActiveSequence] = asyncio.Queue()

        # Metrics
        self.total_cached_tokens = 0
        self.active_count = 0

    async def UpdateWeights(  # noqa: N802
        self,
        request: inference_pb2.WeightUpdateRequest,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.WeightUpdateResponse:
        """Update policy model weights with LoRA matrices."""
        logger.info(f"Weight update request: {len(request.updates)} layers")

        try:
            for update in request.updates:
                layer_idx = update.layer_idx
                param_name = update.param_name

                # Convert bytes to MLX arrays
                # We assume the bytes are bf16 bits, which NumPy can handle as uint16
                # Then we view as bfloat16 in MLX.
                # If the buffer is empty, we handle it.
                if not update.lora_A or not update.lora_B:
                    logger.warning(f"Empty LoRA update for layer {layer_idx}")
                    continue

                a_np = np.frombuffer(update.lora_A, dtype=np.uint16).reshape(list(update.shape_A))
                b_np = np.frombuffer(update.lora_B, dtype=np.uint16).reshape(list(update.shape_B))

                lora_A = mx.array(a_np).view(mx.bfloat16)
                lora_B = mx.array(b_np).view(mx.bfloat16)

                # Compute delta: B @ A
                delta_W = lora_B @ lora_A # (out_dim, rank) @ (rank, in_dim) -> (out_dim, in_dim)

                # Apply to policy model
                # param_name might be "self_attn.q_proj"
                target_layer = self.model.layers[layer_idx]

                if param_name == "self_attn.q_proj":
                    # qkv_proj.weight shape is ( (num_heads + 2*num_kv_heads) * head_dim, dim )
                    # delta_W shape is ( num_heads * head_dim, dim )
                    W = target_layer.self_attn.qkv_proj.weight
                    new_W = mx.concatenate([
                        W[:delta_W.shape[0], :] + delta_W,
                        W[delta_W.shape[0]:, :]
                    ], axis=0)
                    target_layer.self_attn.qkv_proj.weight = new_W
                elif param_name == "self_attn.k_proj":
                    # K part starts after Q
                    q_size = self.num_heads * self.head_dim
                    k_size = self.num_kv_heads * self.head_dim
                    W = target_layer.self_attn.qkv_proj.weight
                    new_W = mx.concatenate([
                        W[:q_size, :],
                        W[q_size : q_size + k_size, :] + delta_W,
                        W[q_size + k_size:, :]
                    ], axis=0)
                    target_layer.self_attn.qkv_proj.weight = new_W
                elif param_name == "self_attn.v_proj":
                    # V part is at the end
                    q_size = self.num_heads * self.head_dim
                    k_size = self.num_kv_heads * self.head_dim
                    W = target_layer.self_attn.qkv_proj.weight
                    new_W = mx.concatenate([
                        W[:q_size + k_size, :],
                        W[q_size + k_size:, :] + delta_W
                    ], axis=0)
                    target_layer.self_attn.qkv_proj.weight = new_W
                elif param_name == "self_attn.qkv_proj":
                    target_layer.self_attn.qkv_proj.weight += delta_W
                elif param_name == "mlp.gate_up_proj":
                    target_layer.mlp.gate_up_proj.weight += delta_W
                elif param_name == "mlp.down_proj":
                    target_layer.mlp.down_proj.weight += delta_W

            self.policy_version += 1
            # Invalidate policy prefix cache because weights changed
            self.prefix_cache = PrefixCache(self.allocator)

            return inference_pb2.WeightUpdateResponse(success=True, policy_version=self.policy_version)

        except Exception as e:
            logger.error(f"Weight update error: {e}")
            return inference_pb2.WeightUpdateResponse(success=False)

    async def Rollout(  # noqa: N802
        self,
        request: inference_pb2.RolloutRequest,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.RolloutResponse:
        """Process GRPO rollout request: generate rollouts + reference log-probs."""
        logger.info(f"Rollout request {request.batch_id}: {len(request.prompts)} prompts, G={request.group_size}")

        batch_id = request.batch_id
        G = request.group_size
        all_results = []

        # Process prompts in chunks to manage memory
        # Automatically scale concurrent prompts based on group size
        max_concurrent_prompts = max(1, self.max_inflight_rollouts // G)
        for i in range(0, len(request.prompts), max_concurrent_prompts):
            chunk_prompts = request.prompts[i : i + max_concurrent_prompts]
            logger.info(f"Processing rollout chunk: prompts {i} to {i + len(chunk_prompts)} (Batch Size: {len(chunk_prompts) * G})")

            chunk_results = await self._process_rollout_chunk(
                chunk_prompts, G, request.temperature, request.max_tokens
            )

            # Phase 2: Reference Model log-probs (Batched)
            logger.info(f"Processing ref log-probs chunk: prompts {i} to {i + len(chunk_prompts)}")
            await self._process_ref_logprobs_batched(chunk_prompts, chunk_results, G)

            all_results.extend(chunk_results)

        return inference_pb2.RolloutResponse(batch_id=batch_id, results=all_results)

    async def Judge(  # noqa: N802
        self,
        request: inference_pb2.JudgeRequest,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.JudgeResponse:
        """Process GRPO judge request: evaluate completions using shared rubric."""
        logger.info(f"Judge request {request.batch_id}: {len(request.items)} items")

        batch_id = request.batch_id
        rubric_tokens = list(request.rubric_tokens)

        # 1. Prefill rubric (shared prefix)
        if rubric_tokens:
            matched_len, matched_blocks = self.prefix_cache.lookup(rubric_tokens)
            rubric_cache = self._create_cache(batch_size=1)
            if matched_len > 0:
                rubric_cache.offsets = mx.array([matched_len], dtype=mx.int32)
                rubric_cache.block_tables[0] = matched_blocks

            rubric_suffix = mx.array(rubric_tokens[matched_len:], dtype=mx.uint32).reshape(1, -1)
            _, rubric_cache = self.model(rubric_suffix, cache=rubric_cache)
            mx.eval(rubric_cache.offsets)
            rubric_cache.advance(len(rubric_tokens) - matched_len)
            self.prefix_cache.insert(rubric_tokens, rubric_cache.block_tables[0])

        results = []
        # Process judge items
        # Rely on prefix caching for efficiency across items sharing prompts
        for item in request.items:
            full_prefix = rubric_tokens + list(item.prompt_tokens)
            matched_len, matched_blocks = self.prefix_cache.lookup(full_prefix)

            cache = self._create_cache(batch_size=1)
            if matched_len > 0:
                cache.offsets = mx.array([matched_len], dtype=mx.int32)
                cache.block_tables[0] = matched_blocks

            suffix = mx.array(full_prefix[matched_len:], dtype=mx.uint32).reshape(1, -1)
            logits, cache = self.model(suffix, cache=cache)
            mx.eval(logits)
            cache.advance(len(full_prefix) - matched_len)
            self.prefix_cache.insert(full_prefix, cache.block_tables[0])

            # Decode verdict tokens
            verdict_tokens = []
            verdict_log_probs = []

            from model import fused_log_softmax

            # Sample first token
            if request.temperature > 0:
                token = self._sample_token(logits[0, -1], request.temperature)
            else:
                token = int(mx.argmax(logits[0, -1]))

            log_probs = fused_log_softmax(logits[0, -1:], request.temperature or 1.0)[0]
            verdict_tokens.append(token)
            verdict_log_probs.append(float(log_probs[token]))

            for _ in range(request.max_tokens - 1):
                if token == self.eos_token_id:
                    break

                input_token = mx.array([[token]], dtype=mx.uint32)
                logits, cache = self.model(input_token, cache=cache)
                mx.eval(logits)
                cache.advance(1)

                log_probs = fused_log_softmax(logits[0, 0], request.temperature or 1.0)
                if request.temperature > 0:
                    token = self._sample_token(logits[0, 0], request.temperature)
                else:
                    token = int(mx.argmax(logits[0, 0]))

                verdict_tokens.append(token)
                verdict_log_probs.append(float(log_probs[token]))

            results.append(inference_pb2.JudgeResult(
                item_id=item.item_id,
                verdict_tokens=verdict_tokens,
                log_probs=verdict_log_probs
            ))

        return inference_pb2.JudgeResponse(batch_id=batch_id, results=results)

    async def _process_ref_logprobs_batched(
        self,
        prompts: list[inference_pb2.Prompt],
        results: list[inference_pb2.PromptRollout],
        G: int,
    ) -> None:
        """Compute reference model log-probs for completions using batching."""
        from model import fused_log_softmax

        for p_idx, (p, p_res) in enumerate(zip(prompts, results)):
            tokens_list = list(p.tokens)
            matched_len, matched_blocks = self.reference_prefix_cache.lookup(tokens_list)

            # 1. Prefill prompt once
            cache = self._create_cache(batch_size=1)
            if matched_len > 0:
                cache.offsets = mx.array([matched_len], dtype=mx.int32)
                cache.block_tables[0] = matched_blocks

            tokens_suffix = mx.array(tokens_list[matched_len:], dtype=mx.uint32).reshape(1, -1)
            logits, cache = self.reference_model(tokens_suffix, cache=cache)
            mx.eval(logits)
            cache.advance(len(tokens_list) - matched_len)
            self.reference_prefix_cache.insert(tokens_list, cache.block_tables[0])

            # 2. Setup batched forward pass for completions
            batched_cache = self._create_cache(batch_size=G)
            for g in range(G):
                batched_cache.block_tables[g] = list(cache.block_tables[0])
                for b in batched_cache.block_tables[g]:
                    self.allocator.retain(b)
                batched_cache.offsets[g] = int(cache.offsets[0])

            # Log-prob for first token
            log_probs_first = fused_log_softmax(logits[0, -1:], 1.0)[0]
            for g in range(G):
                first_token = p_res.completions[g].tokens[0]
                p_res.completions[g].ref_log_probs.append(float(log_probs_first[first_token]))

            max_len = max(len(c.tokens) for c in p_res.completions)
            for step in range(max_len - 1):
                step_tokens = []
                active_mask = []
                for g in range(G):
                    if step < len(p_res.completions[g].tokens) - 1:
                        step_tokens.append(p_res.completions[g].tokens[step])
                        active_mask.append(True)
                    else:
                        step_tokens.append(0)
                        active_mask.append(False)

                step_tokens_mx = mx.array(step_tokens, dtype=mx.uint32).reshape(-1, 1)
                logits_step, _ = self.reference_model(step_tokens_mx, cache=batched_cache)
                mx.eval(logits_step)
                batched_cache.advance(1)

                log_probs_step = fused_log_softmax(logits_step[:, 0, :], 1.0)
                for g in range(G):
                    if active_mask[g]:
                        next_token = p_res.completions[g].tokens[step + 1]
                        p_res.completions[g].ref_log_probs.append(float(log_probs_step[g, next_token]))

    async def _process_rollout_chunk(
        self,
        prompts: list[inference_pb2.Prompt],
        G: int,
        temperature: float,
        max_tokens: int,
    ) -> list[inference_pb2.PromptRollout]:
        """Generate rollouts for a chunk of prompts."""
        num_prompts = len(prompts)
        total_sequences = num_prompts * G
        from model import fused_log_softmax

        # 1. Prefill prompts and collect logits
        prompt_data = [] # (last_logits, block_table, offset, tokens)
        for p in prompts:
            tokens_list = list(p.tokens)
            matched_len, matched_blocks = self.prefix_cache.lookup(tokens_list)
            cache = self._create_cache(batch_size=1)
            if matched_len > 0:
                cache.offsets = mx.array([matched_len], dtype=mx.int32)
                cache.block_tables[0] = matched_blocks

            tokens_suffix = mx.array(tokens_list[matched_len:], dtype=mx.uint32).reshape(1, -1)
            logits, cache = self.model(tokens_suffix, cache=cache)
            mx.eval(logits)
            cache.advance(len(tokens_list) - matched_len)
            self.prefix_cache.insert(tokens_list, cache.block_tables[0])
            prompt_data.append((logits[0, -1], cache.block_tables[0], int(cache.offsets[0]), tokens_list))

        # 2. Fork G rollouts per prompt
        batched_cache = self._create_cache(batch_size=total_sequences)
        active_tokens = [] # (total_sequences,)
        active_log_probs = [] # (total_sequences,)

        for p_idx, (last_logits, p_blocks, p_offset, p_tokens) in enumerate(prompt_data):
            # Compute log_probs for all vocab at once for this prompt's end
            log_probs_all = fused_log_softmax(last_logits[None], temperature or 1.0)[0]

            for g_idx in range(G):
                seq_idx = p_idx * G + g_idx
                # Fork blocks and offset
                batched_cache.block_tables[seq_idx] = list(p_blocks)
                for b in p_blocks:
                    self.allocator.retain(b)
                batched_cache.offsets[seq_idx] = p_offset

                # Sample first token of rollout
                if temperature > 0:
                    token = self._sample_token(last_logits, temperature)
                else:
                    token = int(mx.argmax(last_logits))

                active_tokens.append(token)
                active_log_probs.append(float(log_probs_all[token]))

        # 3. Decode loop
        seq_results = [{"tokens": [active_tokens[i]], "log_probs": [active_log_probs[i]]} for i in range(total_sequences)]
        is_finished = [False] * total_sequences

        for step in range(max_tokens - 1):
            if all(is_finished):
                break

            current_tokens_mx = mx.array(active_tokens, dtype=mx.uint32).reshape(-1, 1)

            logits, batched_cache = self.model(current_tokens_mx, cache=batched_cache)
            mx.eval(logits)

            # Update offsets in cache
            batched_cache.advance(1)

            # log_probs for all tokens in batch: (B, V)
            log_probs_batch = fused_log_softmax(logits[:, 0, :], temperature or 1.0)

            new_active_tokens = []
            for b_idx in range(total_sequences):
                if is_finished[b_idx]:
                    new_active_tokens.append(self.eos_token_id)
                    continue

                # Sample next token
                if temperature > 0:
                    token = self._sample_token(logits[b_idx, 0], temperature)
                else:
                    token = int(mx.argmax(logits[b_idx, 0]))

                seq_results[b_idx]["tokens"].append(token)
                seq_results[b_idx]["log_probs"].append(float(log_probs_batch[b_idx, token]))

                if token == self.eos_token_id or len(seq_results[b_idx]["tokens"]) >= max_tokens:
                    is_finished[b_idx] = True

                new_active_tokens.append(token)

            active_tokens = new_active_tokens
            if step % 10 == 0:
                logger.info(f"Rollout step {step}: {sum(is_finished)}/{total_sequences} finished")

        # 4. Format results
        final_prompt_results = []
        for p_idx in range(num_prompts):
            completions = []
            for g_idx in range(G):
                seq_idx = p_idx * G + g_idx
                res = seq_results[seq_idx]
                completions.append(inference_pb2.Completion(
                    tokens=res["tokens"],
                    log_probs=res["log_probs"]
                ))
            final_prompt_results.append(inference_pb2.PromptRollout(
                prompt_id=prompts[p_idx].prompt_id,
                completions=completions
            ))

        return final_prompt_results


    def _create_cache(self, batch_size: int = 1) -> KVCache:
        """Create a new KV cache instance."""
        return KVCache(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            batch_size=batch_size,
            allocator=self.allocator,
        )

    async def Prefill(  # noqa: N802
        self,
        request: inference_pb2.PrefillRequest,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.PrefillResponse:
        """Process prefill request: populate KV cache and return first token."""
        logger.info(f"Prefill request {request.request_id}: {len(request.tokens)} tokens")

        try:
            num_tokens = len(request.tokens)

            # Check prefix cache
            matched_len, matched_blocks = self.prefix_cache.lookup(request.tokens)
            if matched_len > 0:
                logger.info(f"Prefix cache hit for {request.request_id}: {matched_len} tokens")

            # Convert tokens to MLX array (only suffix needs prefill)
            tokens = mx.array(request.tokens, dtype=mx.uint32).reshape(1, -1)
            tokens_suffix = tokens[:, matched_len:]

            # Create new cache for this sequence
            cache = self._create_cache()
            if matched_len > 0:
                cache.offset = matched_len
                cache.block_tables[0] = matched_blocks

            # Run prefill through model
            t0 = time.perf_counter()
            logits, cache = self.model(tokens_suffix, cache=cache)
            mx.eval(logits)  # Ensure computation is complete

            # The model forward pass already updated the cache with new tokens,
            # but we need to advance the offset for the suffix length.
            # Wait, cache.update already used matched_len as base.
            # The original code called cache.advance(num_tokens).
            # Here, we only need to advance by the suffix length.
            cache.advance(num_tokens - matched_len)

            elapsed = time.perf_counter() - t0
            tps = (num_tokens - matched_len) / elapsed if elapsed > 0 else float("inf")
            logger.info(f"Prefill {request.request_id}: {num_tokens - matched_len} new tokens in {elapsed:.3f}s ({tps:.1f} tok/s)")

            # Insert full prompt into prefix cache
            self.prefix_cache.insert(request.tokens, cache.block_tables[0])

            # Sample next token
            if request.temperature > 0:
                next_token = self._sample_token(logits[0, -1], request.temperature)
            else:
                next_token = int(mx.argmax(logits[0, -1]))

            # Track sequence
            seq = ActiveSequence(
                request_id=request.request_id,
                tokens=[*list(request.tokens), next_token],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                cache=cache,
            )
            self.active_sequences[request.request_id] = seq
            self.total_cached_tokens += num_tokens

            return inference_pb2.PrefillResponse(
                request_id=request.request_id,
                next_token_id=next_token,
                cache_position=num_tokens,
            )

        except Exception as e:
            logger.error(f"Prefill error for {request.request_id}: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Prefill failed: {e}")

    async def Decode(  # noqa: N802
        self,
        request: inference_pb2.DecodeRequest,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.DecodeResponse:
        """Process decode request: generate next token using cached KV."""
        logger.debug(f"Decode request {request.request_id}")

        try:
            seq = self.active_sequences.get(request.request_id)
            if not seq:
                await context.abort(grpc.StatusCode.NOT_FOUND, f"Request {request.request_id} not found")
                return

            # Get last token
            last_token = mx.array(np.array([[seq.tokens[-1]]]), dtype=mx.uint32)

            # Use stored cache
            cache = seq.cache

            # Run single decode step
            t0 = time.perf_counter()
            logits, cache = self.model(last_token, cache=cache)
            mx.eval(logits)
            cache.advance(1)

            elapsed = time.perf_counter() - t0
            tps = 1.0 / elapsed if elapsed > 0 else float("inf")

            # Sample next token
            if seq.temperature > 0:
                next_token = self._sample_token(logits[0, 0], seq.temperature)
            else:
                next_token = int(mx.argmax(logits[0, 0]))

            # Update sequence
            seq.tokens.append(next_token)
            seq.is_finished = len(seq.tokens) >= seq.max_tokens or next_token == self.eos_token_id

            logger.info(f"Decode {request.request_id}: token={next_token} ({tps:.1f} tok/s, finished={seq.is_finished})")

            return inference_pb2.DecodeResponse(
                request_id=request.request_id,
                token_id=next_token,
                is_finished=seq.is_finished,
            )

        except Exception as e:
            logger.error(f"Decode error for {request.request_id}: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Decode failed: {e}")

    async def StreamGenerate(  # noqa: N802
        self,
        request: inference_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[inference_pb2.TokenResponse]:
        """Stream generated tokens for a request."""
        logger.info(f"StreamGenerate request {request.request_id}: {len(request.tokens)} input tokens")

        try:
            num_input_tokens = len(request.tokens)

            # Check prefix cache
            matched_len, matched_blocks = self.prefix_cache.lookup(request.tokens)
            if matched_len > 0:
                logger.info(f"Prefix cache hit for {request.request_id}: {matched_len} tokens")

            # Convert tokens to MLX array (only suffix needs prefill)
            tokens = mx.array(request.tokens, dtype=mx.uint32).reshape(1, -1)
            tokens_suffix = tokens[:, matched_len:]

            # Create new cache for this sequence
            cache = self._create_cache()
            if matched_len > 0:
                cache.offset = matched_len
                cache.block_tables[0] = matched_blocks

            # Run prefill through model
            t0 = time.perf_counter()
            logits, cache = self.model(tokens_suffix, cache=cache)
            mx.eval(logits)

            # Update cache offset by suffix length
            cache.advance(num_input_tokens - matched_len)

            prefill_elapsed = time.perf_counter() - t0
            prefill_tps = (num_input_tokens - matched_len) / prefill_elapsed if prefill_elapsed > 0 else float("inf")
            logger.info(f"StreamGenerate {request.request_id} prefill: {num_input_tokens - matched_len} new tokens in {prefill_elapsed:.3f}s ({prefill_tps:.1f} tok/s)")

            # Insert full prompt into prefix cache
            self.prefix_cache.insert(request.tokens, cache.block_tables[0])

            # Sample first token
            if request.temperature > 0:
                next_token = self._sample_token(logits[0, -1], request.temperature)
            else:
                next_token = int(mx.argmax(logits[0, -1]))

            generated_tokens = [next_token]

            # Yield first token
            yield inference_pb2.TokenResponse(
                request_id=request.request_id,
                token_id=next_token,
                is_finished=False,
            )

            # Generate remaining tokens
            decode_start = time.perf_counter()
            for _ in range(request.max_tokens - 1):
                last_token = mx.array([[generated_tokens[-1]]], dtype=mx.uint32)

                t0 = time.perf_counter()
                logits, cache = self.model(last_token, cache=cache)
                mx.eval(logits)
                cache.advance(1)

                step_elapsed = time.perf_counter() - t0
                step_tps = 1.0 / step_elapsed if step_elapsed > 0 else float("inf")

                if request.temperature > 0:
                    next_token = self._sample_token(logits[0, 0], request.temperature)
                else:
                    next_token = int(mx.argmax(logits[0, 0]))

                generated_tokens.append(next_token)

                is_finished = next_token == self.eos_token_id or len(generated_tokens) >= request.max_tokens

                logger.info(f"StreamGenerate {request.request_id} decode: token {len(generated_tokens)} ({step_tps:.1f} tok/s)")

                yield inference_pb2.TokenResponse(
                    request_id=request.request_id,
                    token_id=next_token,
                    is_finished=is_finished,
                )

                if is_finished:
                    break

            decode_elapsed = time.perf_counter() - decode_start
            num_decoded = len(generated_tokens) - 1  # exclude first token (from prefill)
            decode_tps = num_decoded / decode_elapsed if decode_elapsed > 0 and num_decoded > 0 else 0.0
            logger.info(
                f"StreamGenerate {request.request_id} complete: "
                f"prefill {num_input_tokens} tokens @ {prefill_tps:.1f} tok/s, "
                f"decode {num_decoded} tokens @ {decode_tps:.1f} tok/s"
            )

        except Exception as e:
            logger.error(f"StreamGenerate error for {request.request_id}: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Generation failed: {e}")

    async def GetStatus(  # noqa: N802
        self,
        request: inference_pb2.StatusRequest,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.NodeStatus:
        """Return current node status."""
        return inference_pb2.NodeStatus(
            active_sequences=len(self.active_sequences),
            gpu_utilization=0.0,  # TODO: implement Metal GPU monitoring
            queue_depth=self.sequence_queue.qsize(),
            cached_tokens=self.total_cached_tokens,
        )

    def _sample_token(self, logits: mx.array, temperature: float, top_k: int = 20) -> int:
        """Sample next token from logits with temperature."""
        if temperature == 0:
            return int(mx.argmax(logits))

        scaled_logits = logits / temperature

        # Get indices of top-k highest logits
        top_indices = mx.argsort(scaled_logits)[-top_k:]
        top_logits = scaled_logits[top_indices]

        # Sample from categorical distribution over top-k, map back to vocab index
        sampled = mx.random.categorical(top_logits)
        return int(top_indices[sampled])


def _register_with_scheduler(grpc_port: int, scheduler_url: str) -> None:
    """Register this node with the scheduler via POST /v1/nodes."""
    hostname = socket.gethostname()
    try:
        addr = socket.gethostbyname(hostname)
    except socket.gaierror:
        addr = "http://localhost"
    node_addr = f"{addr}:{grpc_port}"
    body = json.dumps({"addr": node_addr}).encode()
    req = urlrequest.Request(
        f"{scheduler_url}/v1/nodes",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read())
            logger.info(f"Registered with scheduler: {result}")
    except Exception as e:
        logger.warning(f"Failed to register with scheduler at {scheduler_url}: {e}")


async def serve(
    port: int = 50052,
    checkpoint_path: str = "Qwen/Qwen3-0.6B",
    max_inflight_rollouts: int = DEFAULT_MAX_INFLIGHT_ROLLOUTS,
    scheduler_url: str = "http://localhost:8080",
):
    """Start the gRPC inference node server."""
    logger.info(f"Starting MLX inference node for {checkpoint_path}...")

    # Load model
    logger.info(f"Loading model from {checkpoint_path}")

    # Default Qwen3-0.6B configuration
    config = {
        "vocab_size": 151936,
        "dim": 1024,
        "num_layers": 28,
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 3072,
        "max_seq_len": 2048,
        "rope_theta": 1000000.0,
        "eps": 1e-6,
        "tie_word_embeddings": True,
        "use_qk_norm": True,
        "rope_traditional": False,
    }

    model = Qwen3(**config)
    reference_model = Qwen3(**config)
    checkpoint_path = download_qwen3(checkpoint_path)
    load_qwen3_weights(model, checkpoint_path)
    load_qwen3_weights(reference_model, checkpoint_path)

    logger.info("Models loaded successfully")

    # Create gRPC server
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = InferenceNodeServicer(
        model,
        reference_model,
        max_inflight_rollouts=max_inflight_rollouts,
        max_seq_len=config["max_seq_len"],
        vocab_size=config["vocab_size"],
        eos_token_id=151643, # Qwen3 EOS token
    )

    inference_pb2_grpc.add_InferenceNodeServicer_to_server(servicer, server)

    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)

    logger.info(f"Starting server on {listen_addr}")
    await server.start()

    logger.info(f"MLX inference node listening on port {port}")

    _register_with_scheduler(port, scheduler_url)

    await server.wait_for_termination()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MLX inference node with gRPC server for Qwen3-0.6B")
    parser.add_argument("--port", type=int, default=50052, help="gRPC server port (default: 50052)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model checkpoint path",
    )
    parser.add_argument("--max-inflight-rollouts", type=int, default=DEFAULT_MAX_INFLIGHT_ROLLOUTS, help="Maximum total rollouts (sequences) active at once")

    args = parser.parse_args()

    asyncio.run(serve(port=args.port, checkpoint_path=args.checkpoint, max_inflight_rollouts=args.max_inflight_rollouts))
