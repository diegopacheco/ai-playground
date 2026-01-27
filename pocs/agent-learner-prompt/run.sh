#!/bin/bash
cd "$(dirname "$0")"
cargo build --release 2>/dev/null || cargo build --release
if [ $? -eq 0 ]; then
    ./target/release/agent-learner "$@"
else
    echo "Build failed"
    exit 1
fi
