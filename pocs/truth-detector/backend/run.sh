#!/bin/bash

cargo build --release 2>&1
if [ $? -ne 0 ]; then
  echo "Backend build failed"
  exit 1
fi

./target/release/truth-detector-backend &
BACKEND_PID=$!
echo $BACKEND_PID > /tmp/truth-detector-backend.pid
echo "Backend started on http://localhost:3000 (PID: $BACKEND_PID)"
wait $BACKEND_PID
