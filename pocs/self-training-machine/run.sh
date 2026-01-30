#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
trap 'kill $(jobs -p) 2>/dev/null' EXIT
(cd "$SCRIPT_DIR/backend" && cargo build --release && ./target/release/self-training-machine-backend) &
BACKEND_PID=$!
while ! nc -z localhost 8080 2>/dev/null; do
  if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "Backend failed to start"
    exit 1
  fi
  sleep 1
done
echo "Backend started on http://localhost:8080"
(cd "$SCRIPT_DIR/frontend" && bun install && bun run dev) &
FRONTEND_PID=$!
while ! nc -z localhost 3000 2>/dev/null; do
  if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "Frontend failed to start"
    exit 1
  fi
  sleep 1
done
echo "Frontend started on http://localhost:3000"
echo "Self Training Machine is ready!"
wait
