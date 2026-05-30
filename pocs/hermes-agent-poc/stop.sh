#!/usr/bin/env bash
# Stop the Rock Paper Scissors web server
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$DIR/.server.pid"

if [ ! -f "$PIDFILE" ]; then
  echo "No PID file found. Server may not be running."
  # Fallback: try to find any python http.server on port 8000
  PID=$(lsof -ti tcp:"${PORT:-8000}" 2>/dev/null || true)
  if [ -n "$PID" ]; then
    echo "Found process on port ${PORT:-8000} (PID $PID). Killing..."
    kill "$PID"
  fi
  exit 0
fi

PID=$(cat "$PIDFILE")
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "Stopped server (PID $PID)"
else
  echo "Server not running (stale PID $PID)"
fi
rm -f "$PIDFILE"
