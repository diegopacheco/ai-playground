#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR/backend" && cargo build --release 2>&1
cd "$SCRIPT_DIR/backend" && cargo run --release &
BACKEND_PID=$!
echo "$BACKEND_PID" > "$SCRIPT_DIR/.backend.pid"

cd "$SCRIPT_DIR/frontend" && bun install 2>&1
cd "$SCRIPT_DIR/frontend" && bun run dev &
FRONTEND_PID=$!
echo "$FRONTEND_PID" > "$SCRIPT_DIR/.frontend.pid"

echo "Backend PID: $BACKEND_PID (port 3001)"
echo "Frontend PID: $FRONTEND_PID (port 5173)"
echo "Open http://localhost:5173"

wait
