#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Backend Unit Tests ==="
cd "$SCRIPT_DIR/backend"
cargo test 2>&1
echo ""

echo "=== Frontend Unit Tests ==="
cd "$SCRIPT_DIR/frontend"
npx vitest run 2>&1
echo ""

echo "All unit tests passed."
