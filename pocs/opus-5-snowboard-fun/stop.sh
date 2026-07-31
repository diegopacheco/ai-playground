#!/bin/bash
cd "$(dirname "$0")"
PORT="${PORT:-8123}"

if [ -f .server.pid ]; then
  PID="$(cat .server.pid)"
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    for i in $(seq 1 10); do
      kill -0 "$PID" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$PID" 2>/dev/null || true
  fi
  rm -f .server.pid
fi

LEFT="$(lsof -ti tcp:"$PORT" 2>/dev/null || true)"
if [ -n "$LEFT" ]; then
  echo "$LEFT" | xargs kill -9 2>/dev/null || true
fi

echo "alpine carve stopped"
