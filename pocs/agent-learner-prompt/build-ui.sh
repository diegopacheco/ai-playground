#!/bin/bash
set -e
cd "$(dirname "$0")/ui"
bun install
bun run build
echo "UI built successfully. Run: cargo run --release -- --ui"
