#!/bin/bash
cd "$(dirname "$0")"
echo "Building agent-learner..."
cargo build --release
if [ $? -eq 0 ]; then
    echo "Build OK"
    echo "Binary: ./target/release/agent-learner"
else
    echo "Build FAILED"
    exit 1
fi
