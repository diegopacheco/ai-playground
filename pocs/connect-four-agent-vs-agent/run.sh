#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
pkill -f connect-four-backend 2>/dev/null || true
pkill -f "vite preview" 2>/dev/null || true
lsof -ti:8080 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
sleep 1
cd "$SCRIPT_DIR/backend"
cargo build --release
cd "$SCRIPT_DIR/frontend"
bun install
bun run build
cd "$SCRIPT_DIR/backend"
./target/release/connect-four-backend &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"
cd "$SCRIPT_DIR/frontend"
bun run preview --port 3000 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"
echo "$BACKEND_PID" > "$SCRIPT_DIR/.backend.pid"
echo "$FRONTEND_PID" > "$SCRIPT_DIR/.frontend.pid"
echo "Backend running on http://localhost:8080"
echo "Frontend running on http://localhost:3000"
wait
