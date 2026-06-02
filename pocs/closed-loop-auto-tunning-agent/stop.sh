#!/usr/bin/env bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$DIR/.run"

for name in backend frontend; do
  PIDFILE="$RUN/$name.pid"
  if [ -f "$PIDFILE" ]; then
    PID="$(cat "$PIDFILE")"
    if kill -0 "$PID" 2>/dev/null; then
      echo "Stopping $name (pid $PID)"
      kill "$PID" 2>/dev/null
    fi
    rm -f "$PIDFILE"
  fi
done

for port in 8080 5173; do
  PIDS="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
  if [ -n "$PIDS" ]; then
    echo "Freeing port $port"
    kill $PIDS 2>/dev/null || true
  fi
done

echo "Stopped"
