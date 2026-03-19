#!/bin/bash
BASEDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASEDIR/backend" && cargo build --release 2>&1
cd "$BASEDIR/backend" && ./target/release/werewolf-server &
BACKEND_PID=$!
cd "$BASEDIR/frontend" && npm install && npm run dev &
FRONTEND_PID=$!
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "$BACKEND_PID" > /tmp/werewolf-backend.pid
echo "$FRONTEND_PID" > /tmp/werewolf-frontend.pid
echo "Backend: http://localhost:3000"
echo "Frontend: http://localhost:3001"
wait
