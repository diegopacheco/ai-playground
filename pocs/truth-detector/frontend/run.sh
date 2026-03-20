#!/bin/bash

deno install 2>&1
if [ $? -ne 0 ]; then
  echo "Frontend install failed"
  exit 1
fi

deno task dev &
FRONTEND_PID=$!
echo $FRONTEND_PID > /tmp/truth-detector-frontend.pid
echo "Frontend started on http://localhost:5173 (PID: $FRONTEND_PID)"
wait $FRONTEND_PID
