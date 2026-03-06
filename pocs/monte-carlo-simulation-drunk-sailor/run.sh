#!/bin/bash

cd backend
cargo build --release 2>&1
./target/release/drunk-sailor-backend &
BACKEND_PID=$!

cd ../frontend
bun install 2>&1
bun dev &
FRONTEND_PID=$!

echo ""
echo "Backend running on http://localhost:8080 (PID $BACKEND_PID)"
echo "Frontend running on http://localhost:5173 (PID $FRONTEND_PID)"
echo "Press Ctrl+C to stop both"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
