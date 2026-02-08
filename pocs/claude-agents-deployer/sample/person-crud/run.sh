#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/.pids"

cd "$SCRIPT_DIR/backend"
cargo build --release 2>&1
nohup ./target/release/person-crud-backend > "$SCRIPT_DIR/backend.log" 2>&1 &
BACKEND_PID=$!

for i in $(seq 1 30); do
  if curl -s -o /dev/null http://127.0.0.1:8080/persons 2>/dev/null; then
    echo "Backend started with PID $BACKEND_PID"
    break
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "Backend failed to start. Check $SCRIPT_DIR/backend.log"
    exit 1
  fi
  sleep 1
done

cd "$SCRIPT_DIR/frontend"
npm install 2>&1
PORT=3000 nohup npx react-scripts start > "$SCRIPT_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!

for i in $(seq 1 30); do
  if curl -s -o /dev/null http://127.0.0.1:3000 2>/dev/null; then
    echo "Frontend started with PID $FRONTEND_PID"
    break
  fi
  sleep 1
done

echo "$BACKEND_PID" > "$PID_FILE"
echo "$FRONTEND_PID" >> "$PID_FILE"

echo "Backend: http://localhost:8080"
echo "Frontend: http://localhost:3000"
echo "PIDs saved to $PID_FILE"
