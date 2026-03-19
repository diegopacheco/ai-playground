#!/bin/bash
cd backend && go build -o server . && ./server &
BACKEND_PID=$!
echo $BACKEND_PID > /tmp/bell-curves-backend.pid
echo "Backend started on :8080 (PID: $BACKEND_PID)"
cd ../frontend && bun dev &
FRONTEND_PID=$!
echo $FRONTEND_PID > /tmp/bell-curves-frontend.pid
echo "Frontend started on :5174 (PID: $FRONTEND_PID)"
wait
