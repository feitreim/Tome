"""JAX/Flax inference node with gRPC server."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib import request as urlrequest

import grpc
import jax
import jax.numpy as jnp
from device import print_device_info, setup_mesh
from flax import nnx
from kvcache import KVCache
from load_weights import load_olmoe_weights
from model import OLMoE, OLMoEConfig

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from jaxtyping import Array

# Import generated gRPC code
import sys

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
    """gRPC servicer for JAX inference node."""

    def __init__(
        self,
        model: OLMoE,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
        num_devices: int = 1,
    ):
        self.model = model
        self.config = model.config
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # Initialize KV cache
        self.cache = KVCache.new(
            layers=self.config.num_layers,
            max_seq_len=max_seq_len,
            n_heads=self.config.n_heads,
            head_dim=self.config.head_dim,
            dtype=jnp.bfloat16,
        )

        # Active sequences being processed
        self.active_sequences: dict[str, ActiveSequence] = {}
        self.sequence_queue: asyncio.Queue[ActiveSequence] = asyncio.Queue()

        # Metrics
        self.total_cached_tokens = 0
        self.active_count = 0

    async def Prefill(  # noqa: N802
        self,
        request: inference_pb2.PrefillRequest,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.PrefillResponse:
        """Process prefill request: populate KV cache and return first token."""
        logger.info(f"Prefill request {request.request_id}: {len(request.tokens)} tokens")

        try:
            # Convert tokens to JAX array
            tokens = jnp.array(request.tokens, dtype=jnp.uint32)
            tokens = tokens.reshape(1, -1)  # Add batch dimension

            # Run prefill through model
            logits, self.cache = self.model(tokens, cache=self.cache, cur_pos=0)

            # Sample next token
            if request.temperature > 0:
                next_token = self._sample_token(logits[0, -1], request.temperature)
            else:
                next_token = int(jnp.argmax(logits[0, -1]))

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

            logger.info(f"Prefill complete: next_token={next_token}, cache_pos={cache_position}")

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
            last_token = jnp.array([[seq.tokens[-1]]], dtype=jnp.uint32)

            # Run single decode step
            logits, self.cache = self.model(last_token, cache=self.cache, cur_pos=seq.cache_position)

            # Sample next token
            if seq.temperature > 0:
                next_token = self._sample_token(logits[0, 0], seq.temperature)
            else:
                next_token = int(jnp.argmax(logits[0, 0]))

            # Update sequence
            seq.tokens.append(next_token)
            seq.cache_position += 1
            seq.is_finished = len(seq.tokens) >= seq.max_tokens or next_token == self.config.eos_token_id

            logger.debug(f"Decode: next_token={next_token}, finished={seq.is_finished}")

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
            # First do prefill
            tokens = jnp.array(request.tokens, dtype=jnp.uint32).reshape(1, -1)
            logits, self.cache = self.model(tokens, cache=self.cache, cur_pos=0)

            # Sample first token
            if request.temperature > 0:
                next_token = self._sample_token(logits[0, -1], request.temperature)
            else:
                next_token = int(jnp.argmax(logits[0, -1]))

            cache_position = len(request.tokens)
            generated_tokens = [next_token]

            # Yield first token
            yield inference_pb2.TokenResponse(
                request_id=request.request_id,
                token_id=next_token,
                is_finished=False,
            )

            # Generate remaining tokens
            for _ in range(request.max_tokens - 1):
                last_token = jnp.array([[generated_tokens[-1]]], dtype=jnp.uint32)
                logits, self.cache = self.model(last_token, cache=self.cache, cur_pos=cache_position)

                if request.temperature > 0:
                    next_token = self._sample_token(logits[0, 0], request.temperature)
                else:
                    next_token = int(jnp.argmax(logits[0, 0]))

                generated_tokens.append(next_token)
                cache_position += 1

                is_finished = next_token == self.config.eos_token_id or len(generated_tokens) >= request.max_tokens

                yield inference_pb2.TokenResponse(
                    request_id=request.request_id,
                    token_id=next_token,
                    is_finished=is_finished,
                )

                if is_finished:
                    break

            logger.info(f"StreamGenerate complete: {len(generated_tokens)} tokens generated")

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
            gpu_utilization=0.0,  # TODO: implement GPU monitoring
            queue_depth=self.sequence_queue.qsize(),
            cached_tokens=self.total_cached_tokens,
        )

    def _sample_token(self, logits: Array, temperature: float) -> int:
        """Sample next token from logits with temperature."""
        if temperature == 0:
            return int(jnp.argmax(logits))

        # Apply temperature scaling
        scaled_logits = logits / temperature

        # Sample from categorical distribution
        next_token = jax.random.categorical(jax.random.PRNGKey(0), scaled_logits)

        return int(next_token)


def _register_with_scheduler(grpc_port: int, scheduler_url: str) -> None:
    """Register this node with the scheduler via POST /v1/nodes."""
    hostname = socket.gethostname()
    try:
        addr = socket.gethostbyname(hostname)
    except socket.gaierror:
        addr = "127.0.0.1"
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
    port: int = 50051,
    checkpoint_path: str = "allenai/OLMoE-1B-7B-0924",
    num_devices: int = 1,
    max_batch_size: int = 32,
    scheduler_url: str = "http://localhost:8080",
):
    """Start the gRPC inference node server."""
    logger.info("Starting JAX inference node...")

    # Setup device and mesh
    print_device_info()
    mesh = setup_mesh(num_devices=num_devices)

    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    config = OLMoEConfig()

    with jax.set_mesh(mesh):
        model = OLMoE(config, rngs=nnx.Rngs(0))
        load_olmoe_weights(model, checkpoint_path, mesh=mesh if num_devices > 1 else None)

    logger.info("Model loaded successfully")

    # Create gRPC server
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = InferenceNodeServicer(model, max_batch_size=max_batch_size, num_devices=num_devices)

    inference_pb2_grpc.add_InferenceNodeServicer_to_server(servicer, server)

    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)

    logger.info(f"Starting server on {listen_addr}")
    await server.start()

    logger.info(f"Inference node listening on port {port}")

    _register_with_scheduler(port, scheduler_url)

    await server.wait_for_termination()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JAX inference node with gRPC server")
    parser.add_argument("--port", type=int, default=50051, help="gRPC server port")
    parser.add_argument("--checkpoint", type=str, default="allenai/OLMoE-1B-7B-0924", help="Model checkpoint path")
    parser.add_argument("--num-devices", type=int, default=1, help="Number of devices for tensor parallelism")
    parser.add_argument("--max-batch-size", type=int, default=32, help="Maximum batch size")
    parser.add_argument("--scheduler-url", type=str, default="http://localhost:8080", help="Scheduler URL")

    args = parser.parse_args()

    asyncio.run(
        serve(
            port=args.port,
            checkpoint_path=args.checkpoint,
            num_devices=args.num_devices,
            max_batch_size=args.max_batch_size,
            scheduler_url=args.scheduler_url,
        )
    )
