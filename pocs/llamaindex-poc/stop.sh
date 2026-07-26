#!/bin/bash

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

for name in backend frontend; do
  PIDFILE="logs/$name.pid"
  if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
      pkill -P "$PID" 2>/dev/null
      kill "$PID" 2>/dev/null
      echo "stopped $name ($PID)"
    fi
    rm -f "$PIDFILE"
  fi
done

for port in 8077 5199; do
  PIDS=$(lsof -ti tcp:$port 2>/dev/null)
  if [ -n "$PIDS" ]; then
    echo "$PIDS" | xargs kill -9 2>/dev/null
    echo "freed port $port"
  fi
done

echo "stopped"
