#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
if [ -f .server.pid ]; then
  PID="$(cat .server.pid)"
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    echo "stopped MEGA SLOP (pid $PID)"
  fi
  rm -f .server.pid
else
  echo "not running"
fi
