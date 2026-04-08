#!/bin/bash

lsof -ti:8080 | xargs kill 2>/dev/null
lsof -ti:5173 | xargs kill 2>/dev/null

cd backend
cargo build --release 2>&1
./target/release/tetris-backend &
BACKEND_PID=$!
echo "Backend started with PID $BACKEND_PID"
echo $BACKEND_PID > /tmp/tetris-backend.pid
cd ..

cd frontend
bun install
bun dev &
FRONTEND_PID=$!
echo "Frontend started with PID $FRONTEND_PID"
echo $FRONTEND_PID > /tmp/tetris-frontend.pid
cd ..

echo "Tetris is running!"
echo "Frontend: http://localhost:5173"
echo "Backend:  http://localhost:8080"

wait
