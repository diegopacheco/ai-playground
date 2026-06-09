#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ -f .server.pid ]; then
  PID=$(cat .server.pid)
  if kill "$PID" >/dev/null 2>&1; then
    echo "Stopped server (pid $PID)."
  fi
  rm -f .server.pid
fi

pkill -f "vite" >/dev/null 2>&1 || true
echo "Done."
