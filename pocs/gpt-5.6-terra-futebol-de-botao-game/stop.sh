#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if [ -f .server.pid ]; then
  PID=$(cat .server.pid)
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    for i in {1..30}; do
      if ! kill -0 "$PID" 2>/dev/null; then
        break
      fi
      sleep 1
    done
  fi
  rm -f .server.pid
fi
echo "Mesa 12 stopped"
