#!/usr/bin/env bash
set -e
echo "Building optimized release binary with embedded assets..."
cargo build --release
BINARY_PATH="target/release/rad"
if [ -f "$BINARY_PATH" ]; then
    BINARY_SIZE=$(du -h "$BINARY_PATH" | cut -f1)
    echo "Build successful!"
    echo "Binary location: $BINARY_PATH"
    echo "Binary size: $BINARY_SIZE"
    echo "All files from agents/ and skills/ are embedded in the binary"
else
    echo "Build failed - binary not found at $BINARY_PATH"
    exit 1
fi
