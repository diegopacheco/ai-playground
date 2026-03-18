#!/bin/bash
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR" && cargo build --release 2>&1
"$BASE_DIR/target/release/agent-memory-bench"
