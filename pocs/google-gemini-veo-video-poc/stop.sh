#!/bin/bash
ROOT="$(cd "$(dirname "$0")" && pwd)"

for name in backend frontend; do
  PID_FILE="$ROOT/.$name.pid"
  if [ -f "$PID_FILE" ]; then
    PID="$(cat "$PID_FILE")"
    kill "$PID" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "Stopped $name (pid $PID)"
  fi
done

pkill -f "node server.js" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
echo "Done"
