"""Test client for MLX gRPC inference node."""

import asyncio
import sys
from pathlib import Path

import grpc

# Import generated gRPC code
sys.path.insert(0, str(Path(__file__).parent / "generated"))
import inference_pb2
import inference_pb2_grpc


async def test_prefill(stub: inference_pb2_grpc.InferenceNodeStub):
    """Test prefill request."""
    print("\n=== Testing Prefill ===")

    request = inference_pb2.PrefillRequest(
        request_id="test-mlx-001",
        tokens=[166100, 12075, 25, 248],  # Example token IDs (BOS + short prompt)
        temperature=0.7,
        max_tokens=100,
    )

    response = await stub.Prefill(request)
    print(f"Request ID: {response.request_id}")
    print(f"Next token: {response.next_token_id}")
    print(f"Cache position: {response.cache_position}")

    return response


async def test_decode(stub: inference_pb2_grpc.InferenceNodeStub, request_id: str, cache_position: int):
    """Test decode request."""
    print("\n=== Testing Decode ===")

    request = inference_pb2.DecodeRequest(
        request_id=request_id,
        cache_position=cache_position,
    )

    response = await stub.Decode(request)
    print(f"Request ID: {response.request_id}")
    print(f"Next token: {response.token_id}")
    print(f"Is finished: {response.is_finished}")

    return response


async def test_stream_generate(stub: inference_pb2_grpc.InferenceNodeStub):
    """Test streaming generation."""
    print("\n=== Testing StreamGenerate ===")

    request = inference_pb2.GenerateRequest(
        request_id="test-mlx-stream-001",
        tokens=[166100, 12075, 25, 248],  # BOS + short prompt
        temperature=0.7,
        max_tokens=20,
    )

    print("Streaming tokens:")
    async for response in stub.StreamGenerate(request):
        print(f"  Token {response.token_id} (finished: {response.is_finished})")
        if response.is_finished:
            break


async def test_status(stub: inference_pb2_grpc.InferenceNodeStub):
    """Test status request."""
    print("\n=== Testing GetStatus ===")

    request = inference_pb2.StatusRequest()
    response = await stub.GetStatus(request)

    print(f"Active sequences: {response.active_sequences}")
    print(f"GPU utilization: {response.gpu_utilization:.2%}")
    print(f"Queue depth: {response.queue_depth}")
    print(f"Cached tokens: {response.cached_tokens}")


async def main():
    """Run all tests."""
    # Connect to server (MLX node uses port 50052 by default)
    server_address = "localhost:50052"
    print(f"Connecting to MLX inference node at {server_address}...")

    async with grpc.aio.insecure_channel(server_address) as channel:
        stub = inference_pb2_grpc.InferenceNodeStub(channel)

        # Test status
        await test_status(stub)

        # Test prefill
        prefill_response = await test_prefill(stub)

        # Test decode
        await test_decode(stub, prefill_response.request_id, prefill_response.cache_position)

        # Test streaming generation
        await test_stream_generate(stub)

        # Final status
        await test_status(stub)

    print("\n✓ All MLX gRPC tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
