#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "=== Building Backend ==="
cd "$SCRIPT_DIR/backend" && cargo build 2>&1
BACKEND_EXIT=$?
echo "=== Building Frontend ==="
cd "$SCRIPT_DIR/frontend" && bun run build 2>&1
FRONTEND_EXIT=$?
if [ $BACKEND_EXIT -ne 0 ] || [ $FRONTEND_EXIT -ne 0 ]; then
  echo "BUILD FAILED"
  exit 1
fi
echo "BUILD SUCCESS"
