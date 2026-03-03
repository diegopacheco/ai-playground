#!/bin/bash
lsof -ti:8080 | xargs kill -9 2>/dev/null
lsof -ti:5173 | xargs kill -9 2>/dev/null
if [ -f /tmp/memory-backend.pid ]; then
  kill $(cat /tmp/memory-backend.pid) 2>/dev/null
  rm /tmp/memory-backend.pid
fi
if [ -f /tmp/memory-frontend.pid ]; then
  kill $(cat /tmp/memory-frontend.pid) 2>/dev/null
  rm /tmp/memory-frontend.pid
fi
cd backend && cargo build --release 2>&1
cd ..
cd backend && ./target/release/memory-game-backend &
BACKEND_PID=$!
echo $BACKEND_PID > /tmp/memory-backend.pid
echo "Backend started (PID $BACKEND_PID) on http://localhost:8080"
cd ..
cd frontend && bun dev &
FRONTEND_PID=$!
echo $FRONTEND_PID > /tmp/memory-frontend.pid
echo "Frontend started (PID $FRONTEND_PID) on http://localhost:5173"
