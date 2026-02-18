#!/bin/bash
# Quick start script for the Tome scheduler

set -e

echo "=== Tome Scheduler Startup ==="
echo

cd scheduler

# Build if needed
if [ ! -f "target/release/scheduler" ]; then
    echo "Building scheduler (release)..."
    cargo build --release
    echo
fi

PORT="${SCHEDULER_PORT:-8080}"

echo "Starting scheduler on port $PORT..."
echo

cargo run --release
