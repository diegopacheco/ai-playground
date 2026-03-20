#!/bin/bash

if [ -f /tmp/truth-detector-backend.pid ]; then
  PID=$(cat /tmp/truth-detector-backend.pid)
  kill $PID 2>/dev/null
  rm -f /tmp/truth-detector-backend.pid
  echo "Backend stopped (PID: $PID)"
fi

if [ -f /tmp/truth-detector-frontend.pid ]; then
  PID=$(cat /tmp/truth-detector-frontend.pid)
  kill $PID 2>/dev/null
  rm -f /tmp/truth-detector-frontend.pid
  echo "Frontend stopped (PID: $PID)"
fi

lsof -ti:3000 | xargs kill -9 2>/dev/null
lsof -ti:5173 | xargs kill -9 2>/dev/null
echo "Ports 3000 and 5173 cleared"
