#!/bin/bash
cd backend && go build -o auction-server . && ./auction-server &
BACKEND_PID=$!
cd frontend && bun install && bun run dev &
FRONTEND_PID=$!
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "$BACKEND_PID" > /tmp/auction-backend.pid
echo "$FRONTEND_PID" > /tmp/auction-frontend.pid
echo "Backend: http://localhost:3000"
echo "Frontend: http://localhost:5173"
wait
