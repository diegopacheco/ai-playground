#!/bin/bash
cd "$(dirname "$0")/backend" && cargo build --release 2>&1 && ./target/release/werewolf-server &
BACKEND_PID=$!
cd "$(dirname "$0")/frontend" && npm install && npm run dev &
FRONTEND_PID=$!
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "$BACKEND_PID" > /tmp/werewolf-backend.pid
echo "$FRONTEND_PID" > /tmp/werewolf-frontend.pid
echo "Backend: http://localhost:3000"
echo "Frontend: http://localhost:3001"
wait
