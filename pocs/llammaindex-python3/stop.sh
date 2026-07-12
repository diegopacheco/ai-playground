#!/bin/bash
cd "$(dirname "$0")"

if [ -f server.pid ]; then
  PID=$(cat server.pid)
  if kill "$PID" >/dev/null 2>&1; then
    echo "Stopped app (pid $PID)"
  fi
  rm -f server.pid
else
  echo "No server.pid found"
fi

pkill -f "venv/bin/python app.py" >/dev/null 2>&1 || true
