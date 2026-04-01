#!/bin/bash
cd backend
cargo build --release 2>&1
./target/release/retirement-planner &
BACKEND_PID=$!
echo $BACKEND_PID > ../backend.pid
cd ../frontend
npm install 2>&1
npm run dev &
FRONTEND_PID=$!
echo $FRONTEND_PID > ../frontend.pid
echo "Backend PID: $BACKEND_PID (port 8080)"
echo "Frontend PID: $FRONTEND_PID (port 5173)"
echo "PIDs saved to backend.pid and frontend.pid"
wait
