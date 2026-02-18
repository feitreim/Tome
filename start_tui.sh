#!/bin/bash
# Quick start script for the Tome TUI

set -e

echo "=== Tome TUI Startup ==="
echo

cd tui

# Build if needed
if [ ! -f "target/release/tome-tui" ]; then
    echo "Building TUI (release)..."
    cargo build --release
    echo
fi

echo "Starting TUI..."
echo

cargo run --release
