#!/bin/bash
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$BASE_DIR/workspaces"
echo "Building backend..."
(cd "$BASE_DIR/backend" && cargo build --release)
echo "Installing frontend dependencies..."
(cd "$BASE_DIR/frontend" && bun install)
echo "Starting backend..."
(cd "$BASE_DIR/backend" && ./target/release/local-agent-orama-backend) &
echo $! > "$BASE_DIR/.backend.pid"
echo "Starting frontend..."
(cd "$BASE_DIR/frontend" && bun run dev) &
echo $! > "$BASE_DIR/.frontend.pid"
echo "Frontend: http://localhost:5173"
echo "Backend: http://localhost:8080"
wait
