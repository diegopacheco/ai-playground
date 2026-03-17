#!/bin/bash
pkill -f ai-rpg-backend 2>/dev/null
pkill -f "vite" 2>/dev/null
lsof -ti:3000 | xargs kill 2>/dev/null
sleep 1
cd backend && cargo build --release && cd ..
./backend/target/release/ai-rpg-backend &
echo $! > .backend.pid
cd frontend && bun run dev &
echo $! > ../.frontend.pid
cd ..
echo "Backend: http://localhost:3000"
echo "Frontend: http://localhost:5173"
