#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -f .server.pid ]; then
  PID="$(cat .server.pid)"
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    echo "Times 2 stopped (pid $PID)"
  fi
  rm -f .server.pid
else
  pkill -f "node server.js" 2>/dev/null || true
  echo "Times 2 stopped"
fi
