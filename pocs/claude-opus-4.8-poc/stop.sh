#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ -f server.pid ]; then
  PID="$(cat server.pid)"
  if kill "$PID" 2>/dev/null; then
    echo "stopped pid ${PID}"
  fi
  rm -f server.pid
else
  echo "no server.pid found"
fi
