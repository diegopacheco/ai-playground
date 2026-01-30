#!/bin/bash
cd "$(dirname "$0")"

cleanup() {
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

cd backend
cargo build --release
./target/release/prompt-score-backend &
BACKEND_PID=$!

cd ../frontend
bun install
bun run dev &
FRONTEND_PID=$!

echo "Backend running on http://localhost:8080"
echo "Frontend running on http://localhost:3000"
echo "Press Ctrl+C to stop"

wait
