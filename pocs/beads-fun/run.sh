#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
lsof -ti:8080 | xargs kill -9 2>/dev/null
lsof -ti:5173 | xargs kill -9 2>/dev/null
sleep 1
(cd "$SCRIPT_DIR/backend" && cargo run) &
BACKEND_PID=$!
(cd "$SCRIPT_DIR/frontend" && npm run dev) &
FRONTEND_PID=$!
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
