#!/bin/bash

if [ -f /tmp/truth-detector-backend.pid ]; then
  PID=$(cat /tmp/truth-detector-backend.pid)
  kill $PID 2>/dev/null
  rm -f /tmp/truth-detector-backend.pid
  echo "Backend stopped (PID: $PID)"
fi

lsof -ti:3000 | xargs kill -9 2>/dev/null
echo "Port 3000 cleared"
