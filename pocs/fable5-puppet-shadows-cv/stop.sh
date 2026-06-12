#!/bin/bash
cd "$(dirname "$0")"
if [ ! -f app.pid ]; then
  echo "not running"
  exit 0
fi
PID=$(cat app.pid)
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  while kill -0 "$PID" 2>/dev/null; do
    sleep 1
  done
  echo "stopped pid $PID"
else
  echo "not running"
fi
rm -f app.pid
