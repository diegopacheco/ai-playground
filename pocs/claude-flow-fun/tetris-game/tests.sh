#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "=== Running Backend Tests ==="
cd "$SCRIPT_DIR/backend"
cargo test
echo ""
echo "=== Running Frontend Tests ==="
cd "$SCRIPT_DIR/frontend"
if [ ! -d "node_modules" ]; then
    npm install
fi
npm test
echo ""
echo "=== All Tests Passed ==="
