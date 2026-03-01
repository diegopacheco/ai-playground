#!/bin/bash
cd "$(dirname "$0")"

mkdir -p backend/data backend/uploads

cd backend
cargo build --release 2>&1
./target/release/twitter-backend &
BACKEND_PID=$!
cd ..

cd frontend
bun install 2>&1
bun run dev &
FRONTEND_PID=$!
cd ..

echo "$BACKEND_PID" > .pids
echo "$FRONTEND_PID" >> .pids

echo "Backend running on http://localhost:8080 (PID: $BACKEND_PID)"
echo "Frontend running on http://localhost:5173 (PID: $FRONTEND_PID)"
echo "Default login: admin / admin"
