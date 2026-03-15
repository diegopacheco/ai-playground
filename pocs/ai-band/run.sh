#!/bin/bash
cd backend && cargo build 2>&1 && cd ..
cd frontend && npm install 2>&1 && cd ..
cd backend && cargo run &
BACKEND_PID=$!
cd frontend && PORT=3000 npm start &
FRONTEND_PID=$!
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Backend: http://localhost:8080"
echo "Frontend: http://localhost:3000"
wait
