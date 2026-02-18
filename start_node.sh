#!/bin/bash
# Quick start script for MLX inference node (Nanbeige4.1-3B on Apple Silicon)

set -e

echo "=== MLX Inference Node Startup (Nanbeige4.1-3B) ==="
echo

# Check if running on Apple Silicon
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo "Warning: MLX requires Apple Silicon (M1/M2/M3/M4)"
    echo "Current architecture: $ARCH"
    echo "Please use the JAX implementation instead: ./start_node.sh"
    exit 1
fi

# Generate proto files if needed
if [ ! -d "mlx-impl/generated" ]; then
    echo "Generating gRPC code from proto files..."
    uv run mlx-impl/generate_proto.py
    echo
fi

# Parse arguments
PORT="${1:-50052}"

echo "Starting MLX inference node on port $PORT..."
echo

# Start the server
uv run mlx-impl/node.py \
    --port "$PORT" \
    --max-batch-size 32

# Note: Use Ctrl+C to stop the server
