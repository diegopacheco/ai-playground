#!/bin/bash
cd "$(dirname "$0")"

cd frontend
npm install &
INSTALL_PID=$!

cd ../backend
deno run --allow-net server.ts &
BACKEND_PID=$!

wait $INSTALL_PID
cd ../frontend
npm run dev &
FRONTEND_PID=$!

echo "Backend running on http://localhost:8000"
echo "Frontend running on http://localhost:3000"
echo "Press Ctrl+C to stop"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
