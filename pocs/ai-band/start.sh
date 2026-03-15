#!/bin/bash
cd backend && cargo run &
BACKEND_PID=$!
cd frontend && PORT=3000 npm start &
FRONTEND_PID=$!
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Backend: http://localhost:8080"
echo "Frontend: http://localhost:3000"
echo $BACKEND_PID > /tmp/ai-band-backend.pid
echo $FRONTEND_PID > /tmp/ai-band-frontend.pid
wait
