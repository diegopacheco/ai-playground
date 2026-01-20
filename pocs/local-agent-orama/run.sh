#!/bin/bash
mkdir -p workspaces
echo "Building backend..."
cd backend && cargo build --release
cd ..
echo "Installing frontend dependencies..."
cd frontend && bun install
cd ..
echo "Starting backend..."
cd backend && ./target/release/local-agent-orama-backend &
echo $! > ../.backend.pid
cd ..
echo "Starting frontend..."
cd frontend && bun run dev &
echo $! > ../.frontend.pid
cd ..
echo "Frontend: http://localhost:5173"
echo "Backend: http://localhost:8080"
wait
