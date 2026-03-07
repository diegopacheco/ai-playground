#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/operator"

cargo build --release

echo "Built: target/release/sre-agent"
echo "Built: target/release/kovalski"
