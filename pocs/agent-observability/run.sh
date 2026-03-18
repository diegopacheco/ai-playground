#!/bin/bash
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
podman-compose -f "$BASE_DIR/podman-compose.yml" up -d
cd "$BASE_DIR/backend" && cargo build --release 2>&1
"$BASE_DIR/backend/target/release/agent-observability-backend" &
echo $! > "$BASE_DIR/.backend.pid"
cd "$BASE_DIR/frontend" && bun install && bun run dev &
echo $! > "$BASE_DIR/.frontend.pid"
echo "Jaeger UI:  http://localhost:16686"
echo "Backend:    http://localhost:3000"
echo "Frontend:   http://localhost:5173"
