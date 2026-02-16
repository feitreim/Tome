"""MLX inference node with gRPC server for Qwen3-0.6B."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import sys
import time
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib import request as urlrequest

import grpc
import mlx.core as mx
import numpy as np
from kvcache import KVCache
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


@dataclass
class ActiveSequence:
    """Represents an active generation sequence in the batch."""

    request_id: str
    tokens: list[int]
    temperature: float
    max_tokens: int
    cache_position: int
    is_finished: bool = False


class InferenceNodeServicer(inference_pb2_grpc.InferenceNodeServicer):
    """gRPC servicer for MLX inference node."""

    def __init__(
        self,
        model: Qwen3,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
        vocab_size: int = 151936,
        eos_token_id: int = 151645,
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id

        # Initialize KV cache parameters
        self.num_layers = model.num_layers
        self.num_kv_heads = model.num_kv_heads
        self.head_dim = model.head_dim

        # Active sequences being processed
        self.active_sequences: dict[str, ActiveSequence] = {}
        self.sequence_queue: asyncio.Queue[ActiveSequence] = asyncio.Queue()

        # Metrics
        self.total_cached_tokens = 0
        self.active_count = 0

    def _create_cache(self) -> KVCache:
        """Create a new KV cache instance."""
        return KVCache(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
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
            # Convert tokens to MLX array
            tokens = mx.array(request.tokens, dtype=mx.uint32).reshape(1, -1)

            # Create new cache for this sequence
            cache = self._create_cache()

            # Run prefill through model
            t0 = time.perf_counter()
            logits, cache = self.model(tokens, cache=cache, cur_pos=0)
            mx.eval(logits)  # Ensure computation is complete
            elapsed = time.perf_counter() - t0
            tps = num_tokens / elapsed if elapsed > 0 else float("inf")
            logger.info(f"Prefill {request.request_id}: {num_tokens} tokens in {elapsed:.3f}s ({tps:.1f} tok/s)")

            # Sample next token
            if request.temperature > 0:
                next_token = self._sample_token(logits[0, -1], request.temperature)
            else:
                next_token = int(mx.argmax(logits[0, -1]))

            # Track sequence
            cache_position = len(request.tokens)
            seq = ActiveSequence(
                request_id=request.request_id,
                tokens=[*list(request.tokens), next_token],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                cache_position=cache_position,
            )
            self.active_sequences[request.request_id] = seq
            self.total_cached_tokens += cache_position

            return inference_pb2.PrefillResponse(
                request_id=request.request_id,
                next_token_id=next_token,
                cache_position=cache_position,
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

            # Need to recreate cache from scratch for now
            # In production, we'd store the cache with the sequence
            cache = self._create_cache()

            # Run single decode step
            t0 = time.perf_counter()
            logits, cache = self.model(last_token, cache=cache, cur_pos=seq.cache_position)
            mx.eval(logits)
            elapsed = time.perf_counter() - t0
            tps = 1.0 / elapsed if elapsed > 0 else float("inf")

            # Sample next token
            if seq.temperature > 0:
                next_token = self._sample_token(logits[0, 0], seq.temperature)
            else:
                next_token = int(mx.argmax(logits[0, 0]))

            # Update sequence
            seq.tokens.append(next_token)
            seq.cache_position += 1
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

            # First do prefill
            tokens = mx.array(np.array(request.tokens), dtype=mx.uint32).reshape(1, -1)
            cache = self._create_cache()

            t0 = time.perf_counter()
            logits, cache = self.model(tokens, cache=cache, cur_pos=0)
            mx.eval(logits)
            prefill_elapsed = time.perf_counter() - t0
            prefill_tps = num_input_tokens / prefill_elapsed if prefill_elapsed > 0 else float("inf")
            logger.info(f"StreamGenerate {request.request_id} prefill: {num_input_tokens} tokens in {prefill_elapsed:.3f}s ({prefill_tps:.1f} tok/s)")

            # Sample first token
            if request.temperature > 0:
                next_token = self._sample_token(logits[0, -1], request.temperature)
            else:
                next_token = int(mx.argmax(logits[0, -1]))

            cache_position = len(request.tokens)
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
                logits, cache = self.model(last_token, cache=cache, cur_pos=cache_position)
                mx.eval(logits)
                step_elapsed = time.perf_counter() - t0
                step_tps = 1.0 / step_elapsed if step_elapsed > 0 else float("inf")

                if request.temperature > 0:
                    next_token = self._sample_token(logits[0, 0], request.temperature)
                else:
                    next_token = int(mx.argmax(logits[0, 0]))

                generated_tokens.append(next_token)
                cache_position += 1

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
    max_batch_size: int = 32,
    scheduler_url: str = "http://localhost:8080",
):
    """Start the gRPC inference node server."""
    logger.info("Starting MLX inference node for Qwen3-0.6B...")

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
        "tie_word_embeddings": False,
    }

    model = Qwen3(**config)
    checkpoint_path = download_qwen3(checkpoint_path)
    load_qwen3_weights(model, checkpoint_path)

    logger.info("Model loaded successfully")

    # Create gRPC server
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = InferenceNodeServicer(
        model,
        max_batch_size=max_batch_size,
        max_seq_len=config["max_seq_len"],
        vocab_size=config["vocab_size"],
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
    parser.add_argument("--checkpoint", type=str, default="Qwen/Qwen3-0.6B", help="Model checkpoint path")
    parser.add_argument("--max-batch-size", type=int, default=32, help="Maximum batch size")

    args = parser.parse_args()

    asyncio.run(serve(port=args.port, checkpoint_path=args.checkpoint, max_batch_size=args.max_batch_size))
