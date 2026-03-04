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

logger = logging.getLogger("tome_node")

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
        max_rollout_batch_size: int = 64,
        max_judge_batch_size: int = 32,
        max_ref_batch_size: int = 32,
        max_seq_len: int = 2048,
        vocab_size: int = 166144,
        eos_token_id: int = 166101,
        num_blocks: int = 512,  # Default to ~7.5GB KV cache for Qwen3-0.6B
        block_size: int = 128,
    ):
        self.model = model  # Policy model (active)
        self.reference_model = reference_model  # Reference model (frozen)
        self.max_rollout_batch_size = max_rollout_batch_size
        self.max_judge_batch_size = max_judge_batch_size
        self.max_ref_batch_size = max_ref_batch_size
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
        self.num_heads = model.layers[0].self_attn.num_heads

    async def UpdateWeights(  # noqa: N802
        self,
        request: inference_pb2.WeightUpdateRequest,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.WeightUpdateResponse:
        """Update policy model weights with LoRA matrices.

        Uses reference_model weights as the frozen base, then sets:
            policy_W = base_W + lora_scale * (B @ A)
        This is non-cumulative — each call replaces the previous LoRA effect.
        """
        logger.info(f"Weight update request: {len(request.updates)} layers")

        try:
            # Group updates by layer to batch qkv slicing
            layer_deltas: dict[int, dict[str, tuple[mx.array, float]]] = {}
            for update in request.updates:
                if not update.lora_A or not update.lora_B:
                    logger.warning(f"Empty LoRA update for layer {update.layer_idx}")
                    continue

                a_np = np.frombuffer(update.lora_A, dtype=np.uint16).reshape(list(update.shape_A))
                b_np = np.frombuffer(update.lora_B, dtype=np.uint16).reshape(list(update.shape_B))
                lora_A = mx.array(a_np).view(mx.bfloat16)
                lora_B = mx.array(b_np).view(mx.bfloat16)
                scale = update.lora_scale if update.lora_scale != 0.0 else 1.0
                delta_W = scale * (lora_B @ lora_A)

                layer_deltas.setdefault(update.layer_idx, {})[update.param_name] = delta_W

            for layer_idx, deltas in layer_deltas.items():
                ref_layer = self.reference_model.layers[layer_idx]
                target_layer = self.model.layers[layer_idx]

                if "self_attn.qkv_proj" in deltas:
                    base_W = ref_layer.self_attn.qkv_proj.weight
                    target_layer.self_attn.qkv_proj.weight = base_W + deltas["self_attn.qkv_proj"]
                elif any(k in deltas for k in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")):
                    base_W = ref_layer.self_attn.qkv_proj.weight
                    q_size = self.num_heads * self.head_dim
                    k_size = self.num_kv_heads * self.head_dim
                    q_delta = deltas.get("self_attn.q_proj", mx.zeros_like(base_W[:q_size]))
                    k_delta = deltas.get("self_attn.k_proj", mx.zeros_like(base_W[q_size:q_size + k_size]))
                    v_delta = deltas.get("self_attn.v_proj", mx.zeros_like(base_W[q_size + k_size:]))
                    target_layer.self_attn.qkv_proj.weight = base_W + mx.concatenate([q_delta, k_delta, v_delta], axis=0)

                if "mlp.gate_up_proj" in deltas:
                    target_layer.mlp.gate_up_proj.weight = ref_layer.mlp.gate_up_proj.weight + deltas["mlp.gate_up_proj"]
                if "mlp.down_proj" in deltas:
                    target_layer.mlp.down_proj.weight = ref_layer.mlp.down_proj.weight + deltas["mlp.down_proj"]

            mx.eval(self.model.parameters())
            self.policy_version += 1
            if hasattr(self, 'prefix_cache') and self.prefix_cache is not None:
                self.prefix_cache.clear()
            self.prefix_cache = PrefixCache(self.allocator)

            logger.info(f"Weight update applied: policy_version={self.policy_version}")
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
        try:
            logger.info(f"Rollout request {request.batch_id}: {len(request.prompts)} prompts, G={request.group_size}")

            batch_id = request.batch_id
            G = request.group_size
            all_results = []

            # Process prompts in chunks to manage memory
            # Automatically scale concurrent prompts based on rollout batch size
            max_concurrent_prompts = max(1, self.max_rollout_batch_size // G)
            for i in range(0, len(request.prompts), max_concurrent_prompts):
                chunk_prompts = request.prompts[i : i + max_concurrent_prompts]
                logger.info(f"Processing rollout chunk: prompts {i} to {i + len(chunk_prompts)} (Batch Size: {len(chunk_prompts) * G})")

                t0 = time.perf_counter()
                chunk_results = await self._process_rollout_chunk(
                    chunk_prompts, G, request.temperature, request.max_tokens
                )
                t1 = time.perf_counter()

                # Calculate tokens generated
                num_rollout_tokens = sum(len(res.completions[g].tokens) for res in chunk_results for g in range(G))
                elapsed = t1 - t0
                tps = num_rollout_tokens / elapsed if elapsed > 0 else 0
                logger.info(f"Rollout chunk completed: {num_rollout_tokens} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)")

                # Phase 2: Reference Model log-probs (Batched)
                logger.info(f"Processing ref log-probs chunk: prompts {i} to {i + len(chunk_prompts)}")
                t0_ref = time.perf_counter()
                await self._process_ref_logprobs_batched(chunk_prompts, chunk_results, G, request.temperature)
                t1_ref = time.perf_counter()
                num_ref_tokens = sum(len(res.completions[g].tokens) for res in chunk_results for g in range(G))
                elapsed_ref = t1_ref - t0_ref
                tps_ref = num_ref_tokens / elapsed_ref if elapsed_ref > 0 else 0
                logger.info(f"Ref log-probs chunk completed: {num_ref_tokens} tokens in {elapsed_ref:.2f}s ({tps_ref:.1f} tok/s)")

                all_results.extend(chunk_results)

            return inference_pb2.RolloutResponse(batch_id=batch_id, results=all_results)
        except Exception as e:
            logger.exception(f"Rollout error: {e}")
            if context:
                await context.abort(grpc.StatusCode.INTERNAL, str(e))
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
            if matched_len == len(rubric_tokens) and len(rubric_tokens) > 0:
                matched_len -= 1

            rubric_cache = self._create_cache(batch_size=1)
            if matched_len > 0:
                rubric_cache.offsets = mx.array([matched_len], dtype=mx.int32)
                rubric_cache.block_tables[0] = list(matched_blocks)
                for b in matched_blocks:
                    self.allocator.retain(b)

            rubric_suffix = mx.array(rubric_tokens[matched_len:], dtype=mx.uint32).reshape(1, -1)
            _, rubric_cache = self.model(rubric_suffix, cache=rubric_cache)
            mx.eval(rubric_cache.offsets)
            rubric_cache.advance(len(rubric_tokens) - matched_len)
            self.prefix_cache.insert(rubric_tokens, rubric_cache.block_tables[0])

        t0_judge = time.perf_counter()

        # Process judge items sequentially for prefill, then batch decode
        item_caches = []
        item_logits = []

        t_prefill_start = time.perf_counter()
        for idx, item in enumerate(request.items):
            full_prefix = rubric_tokens + list(item.prompt_tokens)
            matched_len, matched_blocks = self.prefix_cache.lookup(full_prefix)
            if matched_len == len(full_prefix) and len(full_prefix) > 0:
                matched_len -= 1

            cache = self._create_cache(batch_size=1)
            if matched_len > 0:
                cache.offsets = mx.array([matched_len], dtype=mx.int32)
                cache.block_tables[0] = list(matched_blocks)
                for b in matched_blocks:
                    self.allocator.retain(b)

            suffix = mx.array(full_prefix[matched_len:], dtype=mx.uint32).reshape(1, -1)
            logits, cache = self.model(suffix, cache=cache)
            mx.eval(logits)
            cache.advance(len(full_prefix) - matched_len)
            self.prefix_cache.insert(full_prefix, cache.block_tables[0])

            # Flush to pool so we can read it into contiguous array later if needed,
            # but we already have it in cache._keys. We need to retain it.
            item_caches.append(cache)
            item_logits.append(logits[0, -1])

        t_prefill_end = time.perf_counter()
        logger.info(f"Judge prefill for {len(request.items)} items took {t_prefill_end - t_prefill_start:.2f}s")

        # 3. Batched Decode for all verdicts (Chunked for memory efficiency)
        final_judge_results = []
        total_verdict_tokens = 0
        num_items = len(request.items)
        from model import fused_log_softmax

        for chunk_start in range(0, num_items, self.max_judge_batch_size):
            chunk_end = min(chunk_start + self.max_judge_batch_size, num_items)
            chunk_items = request.items[chunk_start:chunk_end]
            chunk_item_caches = item_caches[chunk_start:chunk_end]
            chunk_item_logits = item_logits[chunk_start:chunk_end]
            B_chunk = len(chunk_items)

            # Build batched cache for decoding this chunk
            t_build_cache_start = time.perf_counter()
            batched_cache = self._create_cache(batch_size=B_chunk)
            max_len = max(int(c.offsets[0]) for c in chunk_item_caches)

            H = self.num_kv_heads
            D = self.head_dim
            NL = batched_cache.num_layers
            dtype = chunk_item_caches[0]._keys[0].dtype

            for l in range(NL):
                k_parts = []
                v_parts = []
                for c in chunk_item_caches:
                    k = c._keys[l]
                    v = c._values[l]
                    pad = max_len - k.shape[2]
                    if pad > 0:
                        k = mx.concatenate([k, mx.zeros((1, H, pad, D), dtype=dtype)], axis=2)
                        v = mx.concatenate([v, mx.zeros((1, H, pad, D), dtype=dtype)], axis=2)
                    k_parts.append(k)
                    v_parts.append(v)
                batched_cache._keys[l] = mx.concatenate(k_parts, axis=0)
                batched_cache._values[l] = mx.concatenate(v_parts, axis=0)

            for b in range(B_chunk):
                batched_cache.offsets[b] = int(chunk_item_caches[b].offsets[0])

            mx.eval(batched_cache._keys + batched_cache._values, batched_cache.offsets)
            t_build_cache_end = time.perf_counter()
            logger.info(f"Judge chunk {chunk_start//self.max_judge_batch_size} cache build took {t_build_cache_end - t_build_cache_start:.2f}s")

            t_decode_start = time.perf_counter()
            active_tokens = []
            active_log_probs = [[] for _ in range(B_chunk)]
            chunk_results = [[] for _ in range(B_chunk)]

            # Sample first tokens
            logits_first = mx.stack(chunk_item_logits) # Shape: [B_chunk, V]
            log_probs_first = fused_log_softmax(logits_first.astype(mx.float32), request.temperature or 1.0)

            for i, logits_i in enumerate(chunk_item_logits):
                if request.temperature > 0:
                    token = self._sample_token(logits_i, request.temperature)
                else:
                    token = int(mx.argmax(logits_i))

                active_tokens.append(token)
                active_log_probs[i].append(float(log_probs_first[i, token]))
                chunk_results[i].append(token)
                total_verdict_tokens += 1

            active_mask = [True] * B_chunk

            for _ in range(request.max_tokens - 1):
                if not any(active_mask):
                    break

                input_tokens = mx.array([[t if active_mask[i] else 0] for i, t in enumerate(active_tokens)], dtype=mx.uint32)
                logits_step, _ = self.model(input_tokens, cache=batched_cache)
                mx.eval(logits_step)
                batched_cache.advance(1)

                logits_b = logits_step[:, 0, :]
                log_probs_b = fused_log_softmax(logits_b.astype(mx.float32), request.temperature or 1.0)

                for i in range(B_chunk):
                    if not active_mask[i]:
                        continue

                    logits_i = logits_b[i]
                    if request.temperature > 0:
                        token = self._sample_token(logits_i, request.temperature)
                    else:
                        token = int(mx.argmax(logits_i))

                    active_tokens[i] = token
                    if token == self.eos_token_id:
                        active_mask[i] = False
                    else:
                        chunk_results[i].append(token)
                        active_log_probs[i].append(float(log_probs_b[i, token]))
                        total_verdict_tokens += 1

            t_decode_end = time.perf_counter()
            logger.info(f"Judge chunk {chunk_start//self.max_judge_batch_size} decode took {t_decode_end - t_decode_start:.2f}s")

            for i, item in enumerate(chunk_items):
                final_judge_results.append(inference_pb2.JudgeResult(
                    item_id=item.item_id,
                    verdict_tokens=chunk_results[i],
                    log_probs=active_log_probs[i]
                ))

        elapsed_judge = time.perf_counter() - t0_judge
        tps_judge = total_verdict_tokens / elapsed_judge if elapsed_judge > 0 else 0
        logger.info(f"Judge request completed: {num_items} items, {total_verdict_tokens} tokens in {elapsed_judge:.2f}s ({tps_judge:.1f} tok/s)")

        return inference_pb2.JudgeResponse(batch_id=batch_id, results=final_judge_results)
    async def _process_ref_logprobs_batched(
        self,
        prompts: list[inference_pb2.Prompt],
        results: list[inference_pb2.PromptRollout],
        G: int,
        temperature: float = 1.0,
    ) -> None:
        """Compute reference model log-probs fully batched without KV cache for maximum throughput."""

        def local_compute_logprobs(logits: mx.array, ids: mx.array, temp: float) -> mx.array:
            logits = (logits.astype(mx.float32) / temp)
            log_z = mx.logsumexp(logits, axis=-1, keepdims=True)
            token_logits = mx.take_along_axis(logits, ids[..., None], axis=-1).squeeze(-1)
            return token_logits - log_z.squeeze(-1)

        t0 = time.perf_counter()

        # Build list of all sequences to process
        all_work = []
        for p_idx, p in enumerate(prompts):
            p_toks = list(p.tokens)
            for g in range(G):
                c_toks = list(results[p_idx].completions[g].tokens)
                all_work.append((p_idx, g, p_toks, c_toks))

        num_toks = 0

        # Process in chunks of max_ref_batch_size
        for chunk_start in range(0, len(all_work), self.max_ref_batch_size):
            chunk_end = min(chunk_start + self.max_ref_batch_size, len(all_work))
            chunk = all_work[chunk_start:chunk_end]

            all_seqs = []
            all_targets = []
            seq_lens = []

            for p_idx, g, p_toks, c_toks in chunk:
                if not c_toks:
                    all_seqs.append(p_toks)
                    all_targets.append([])
                    seq_lens.append(len(p_toks))
                    continue

                full_seq = p_toks + c_toks[:-1]
                all_seqs.append(full_seq)
                all_targets.append(c_toks)
                seq_lens.append(len(full_seq))

            max_len = max(seq_lens)
            padded_seqs = [seq + [0] * (max_len - len(seq)) for seq in all_seqs]
            full_ids = mx.array(padded_seqs, dtype=mx.uint32)

            logits, _ = self.reference_model(full_ids, cache=None)

            lazy_lps = []
            for i, (p_idx, g, p_toks, c_toks) in enumerate(chunk):
                if c_toks:
                    p_len = len(p_toks)
                    c_logits = logits[i, p_len - 1 : p_len - 1 + len(c_toks)]
                    c_targets = mx.array(c_toks, dtype=mx.uint32)
                    lps = local_compute_logprobs(c_logits, c_targets, temperature or 1.0)
                    lazy_lps.append((p_idx, g, lps))
                    num_toks += len(c_toks)

            if lazy_lps:
                mx.eval(*[lp for _, _, lp in lazy_lps])
                for p_idx, g, lps in lazy_lps:
                    results[p_idx].completions[g].ref_log_probs.extend(lps.tolist())

        elapsed = time.perf_counter() - t0
        logger.info(f"Ref log-probs non-KV batched completed: {num_toks} tokens in {elapsed:.2f}s ({num_toks/elapsed:.1f} tok/s)")
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

        # 1. Batched Prefill for all prompts in the chunk
        prompt_data = [] # (last_logit, cache, tokens)

        # Parallel prefill in chunks
        for start_idx in range(0, num_prompts, 8):
            end_idx = min(start_idx + 8, num_prompts)
            chunk_prompts = prompts[start_idx:end_idx]
            chunk_size = len(chunk_prompts)

            lookups = [self.prefix_cache.lookup(list(p.tokens)) for p in chunk_prompts]
            matched_lens = [l[0] for l in lookups]
            matched_blocks_list = [l[1] for l in lookups]

            chunk_cache = self._create_cache(batch_size=chunk_size)
            chunk_cache.offsets = mx.array(matched_lens, dtype=mx.int32)
            for i, blocks in enumerate(matched_blocks_list):
                chunk_cache.block_tables[i] = list(blocks)

            if max(matched_lens) > 0:
                for l in range(chunk_cache.num_layers):
                    k, v = chunk_cache.gather_kv(l)
                    chunk_cache._keys[l] = k
                    chunk_cache._values[l] = v

            all_p_tokens = [list(p.tokens) for p in chunk_prompts]
            suffixes = [t[ml:] for t, ml in zip(all_p_tokens, matched_lens)]
            max_s_len = max(len(s) for s in suffixes)
            padded_s = [s + [0] * (max_s_len - len(s)) for s in suffixes]
            suffix_lens = [len(s) for s in suffixes]

            logits, chunk_cache = self.model(mx.array(padded_s, dtype=mx.uint32), cache=chunk_cache)
            mx.eval(logits)

            chunk_cache.advance(mx.array(suffix_lens, dtype=mx.int32))
            chunk_cache.flush_to_pool()
            mx.eval(self.allocator.k_pool + self.allocator.v_pool)

            for i in range(chunk_size):
                p_len = int(chunk_cache.offsets[i])
                last_logit = logits[i, suffix_lens[i] - 1]

                # Persistence
                self.prefix_cache.insert(all_p_tokens[i], chunk_cache.block_tables[i])

                # Individual cache for repeating
                p_cache = self._create_cache(batch_size=1)
                p_cache.offsets = mx.array([p_len], dtype=mx.int32)
                p_cache.block_tables[0] = list(chunk_cache.block_tables[i])
                for l in range(p_cache.num_layers):
                    p_cache._keys[l] = chunk_cache._keys[l][i:i+1, :, :p_len, :]
                    p_cache._values[l] = chunk_cache._values[l][i:i+1, :, :p_len, :]

                prompt_data.append((last_logit, p_cache, all_p_tokens[i]))

        # 2. Fork G rollouts per prompt — copy prefill KV into batched decode cache
        batched_cache = self._create_cache(batch_size=total_sequences)
        active_tokens = []
        active_log_probs = []

        # Build batched KV by repeating each prompt's KV G times, padded to max prompt length
        max_prompt_len = max(int(p_cache.offsets[0]) for _, p_cache, _ in prompt_data)
        H = self.num_kv_heads
        D = self.head_dim
        NL = batched_cache.num_layers
        dtype = prompt_data[0][1]._keys[0].dtype
        kv_parts_k = [[] for _ in range(NL)]
        kv_parts_v = [[] for _ in range(NL)]
        for p_idx, (last_logits, p_cache, p_tokens) in enumerate(prompt_data):
            p_len = p_cache._keys[0].shape[2]
            pad = max_prompt_len - p_len
            for l in range(NL):
                k = p_cache._keys[l]
                v = p_cache._values[l]
                if pad > 0:
                    k = mx.concatenate([k, mx.zeros((1, H, pad, D), dtype=dtype)], axis=2)
                    v = mx.concatenate([v, mx.zeros((1, H, pad, D), dtype=dtype)], axis=2)
                kv_parts_k[l].append(mx.repeat(k, G, axis=0))
                kv_parts_v[l].append(mx.repeat(v, G, axis=0))
        for l in range(NL):
            batched_cache._keys[l] = mx.concatenate(kv_parts_k[l], axis=0)
            batched_cache._values[l] = mx.concatenate(kv_parts_v[l], axis=0)
        mx.eval(batched_cache._keys + batched_cache._values)

        for p_idx, (last_logits, p_cache, p_tokens) in enumerate(prompt_data):
            log_probs_all = fused_log_softmax(last_logits[None].astype(mx.float32), temperature or 1.0)[0]
            p_offset = int(p_cache.offsets[0])

            for g_idx in range(G):
                seq_idx = p_idx * G + g_idx
                batched_cache.offsets[seq_idx] = p_offset

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
            log_probs_batch = fused_log_softmax(logits[:, 0, :].astype(mx.float32), temperature or 1.0)

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
            if (step + 1) % 5 == 0 or step == 0 or (step + 1) == (max_tokens - 1):
                logger.info(f"Rollout step {step + 1}/{max_tokens}: {sum(is_finished)}/{total_sequences} finished")

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
            if matched_len == len(request.tokens) and len(request.tokens) > 0:
                matched_len -= 1

            if matched_len > 0:
                logger.info(f"Prefix cache hit for {request.request_id}: {matched_len} tokens")

            # Convert tokens to MLX array (only suffix needs prefill)
            tokens = mx.array(request.tokens, dtype=mx.uint32).reshape(1, -1)
            tokens_suffix = tokens[:, matched_len:]

            # Create new cache for this sequence
            cache = self._create_cache()
            if matched_len > 0:
                cache.offsets = mx.array([matched_len], dtype=mx.int32)
                cache.block_tables[0] = list(matched_blocks)
                for b in matched_blocks:
                    self.allocator.retain(b)

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
            if matched_len == len(request.tokens) and len(request.tokens) > 0:
                matched_len -= 1

            if matched_len > 0:
                logger.info(f"Prefix cache hit for {request.request_id}: {matched_len} tokens")

            # Convert tokens to MLX array (only suffix needs prefill)
            tokens = mx.array(request.tokens, dtype=mx.uint32).reshape(1, -1)
            tokens_suffix = tokens[:, matched_len:]

            # Create new cache for this sequence
            cache = self._create_cache()
            if matched_len > 0:
                cache.offsets = mx.array([matched_len], dtype=mx.int32)
                cache.block_tables[0] = list(matched_blocks)
                for b in matched_blocks:
                    self.allocator.retain(b)

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
        top_indices = mx.argpartition(scaled_logits, scaled_logits.size - top_k)[-top_k:]
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
    max_rollout_batch_size: int = 64,
    max_judge_batch_size: int = 32,
    max_ref_batch_size: int = 32,
    num_blocks: int = 512,
    block_size: int = 128,
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
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[("grpc.max_receive_message_length", 64 * 1024 * 1024)],
    )
    servicer = InferenceNodeServicer(
        model,
        reference_model,
        max_rollout_batch_size=max_rollout_batch_size,
        max_judge_batch_size=max_judge_batch_size,
        max_ref_batch_size=max_ref_batch_size,
        max_seq_len=config["max_seq_len"],
        vocab_size=config["vocab_size"],
        eos_token_id=151643, # Qwen3 EOS token
        num_blocks=num_blocks,
        block_size=block_size,
    )

    inference_pb2_grpc.add_InferenceNodeServicer_to_server(servicer, server)

    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)

    logger.info(f"Starting server on {listen_addr}")
    await server.start()

    logger.info(f"MLX inference node listening on port {port}")

    _register_with_scheduler(port, scheduler_url)

    await server.wait_for_termination()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MLX inference node with gRPC server for Qwen3-0.6B")
    parser.add_argument("--port", type=int, default=50052, help="gRPC server port (default: 50052)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model checkpoint path",
    )
    parser.add_argument("--max-rollout-batch-size", type=int, default=64, help="Maximum total rollouts (sequences) active at once during generation")
    parser.add_argument("--max-judge-batch-size", type=int, default=32, help="Maximum total judge items active at once during decoding")
    parser.add_argument("--max-ref-batch-size", type=int, default=32, help="Maximum total ref sequences active at once")
    parser.add_argument("--num-blocks", type=int, default=512, help="Number of KV blocks in the pool")
    parser.add_argument("--block-size", type=int, default=128, help="Number of tokens per KV block")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (can be used multiple times, -v for INFO, -vv for DEBUG)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging (equivalent to -vv)")

    args = parser.parse_args()

    print(f"max rollouts concurrently: {args.max_rollout_batch_size}")
    print(f"max ref lp calc concurrently: {args.max_ref_batch_size}")
    print(f"max judge concurrently: {args.max_judge_batch_size}")


    # Configure logging based on verbosity
    log_level = logging.WARNING
    if args.debug or args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        # Default to INFO if neither -v nor --debug is provided, to match previous behavior
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True
    )
    logger.setLevel(log_level)
    logger.info(f"Logging initialized at level {logging.getLevelName(log_level)}")

    asyncio.run(serve(
        port=args.port,
        checkpoint_path=args.checkpoint,
        max_rollout_batch_size=args.max_rollout_batch_size,
        max_judge_batch_size=args.max_judge_batch_size,
        max_ref_batch_size=args.max_ref_batch_size,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
    ))

if __name__ == "__main__":
    main()
