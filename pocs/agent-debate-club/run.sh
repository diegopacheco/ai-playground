#!/bin/bash
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR/backend" && cargo build --release 2>&1
"$BASE_DIR/backend/target/release/debate-club-backend" &
echo $! > "$BASE_DIR/.backend.pid"
cd "$BASE_DIR/frontend" && bun run dev &
echo $! > "$BASE_DIR/.frontend.pid"
echo "Backend running on http://localhost:3000"
echo "Frontend running on http://localhost:5173"
