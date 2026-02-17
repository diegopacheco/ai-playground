#!/bin/bash

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    if [ -n "$BACKEND_PID" ]; then
        kill "$BACKEND_PID" 2>/dev/null
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill "$FRONTEND_PID" 2>/dev/null
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

(cd backend && cargo build --release && ./target/release/backend) &
BACKEND_PID=$!

(cd frontend && npm install && npm run dev) &
FRONTEND_PID=$!

while ! curl -s http://localhost:8080 > /dev/null 2>&1; do
    sleep 1
done
echo "Backend: http://localhost:8080"

while ! curl -s http://localhost:3000 > /dev/null 2>&1; do
    sleep 1
done
echo "Frontend: http://localhost:3000"

wait
