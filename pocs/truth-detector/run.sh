#!/bin/bash

DIR=$(cd "$(dirname "$0")" && pwd)

cd "$DIR/backend"
cargo build --release 2>&1
if [ $? -ne 0 ]; then
  echo "Backend build failed"
  exit 1
fi

./target/release/truth-detector-backend &
BACKEND_PID=$!
echo $BACKEND_PID > /tmp/truth-detector-backend.pid
echo "Backend started on http://localhost:3000 (PID: $BACKEND_PID)"

cd "$DIR/frontend"
deno install 2>&1
if [ $? -ne 0 ]; then
  echo "Frontend install failed"
  exit 1
fi

deno task dev &
FRONTEND_PID=$!
echo $FRONTEND_PID > /tmp/truth-detector-frontend.pid
echo "Frontend started on http://localhost:5173 (PID: $FRONTEND_PID)"

echo "Truth Detector running"
echo "  Backend:  http://localhost:3000"
echo "  Frontend: http://localhost:5173"

wait
