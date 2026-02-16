#!/bin/bash
cd backend && cargo run &
BACKEND_PID=$!
cd frontend && npm run dev &
FRONTEND_PID=$!
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
