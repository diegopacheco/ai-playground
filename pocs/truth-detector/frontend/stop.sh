#!/bin/bash

if [ -f /tmp/truth-detector-frontend.pid ]; then
  PID=$(cat /tmp/truth-detector-frontend.pid)
  kill $PID 2>/dev/null
  rm -f /tmp/truth-detector-frontend.pid
  echo "Frontend stopped (PID: $PID)"
fi

lsof -ti:5173 | xargs kill -9 2>/dev/null
echo "Port 5173 cleared"
