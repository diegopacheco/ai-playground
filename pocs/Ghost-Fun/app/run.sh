#!/bin/bash
cd backend && cargo build --release 2>&1
cd ..
cd backend && ./target/release/memory-game-backend &
BACKEND_PID=$!
echo $BACKEND_PID > /tmp/memory-backend.pid
echo "Backend started (PID $BACKEND_PID) on http://localhost:8080"
cd frontend && bun dev &
FRONTEND_PID=$!
echo $FRONTEND_PID > /tmp/memory-frontend.pid
echo "Frontend started (PID $FRONTEND_PID) on http://localhost:5173"
