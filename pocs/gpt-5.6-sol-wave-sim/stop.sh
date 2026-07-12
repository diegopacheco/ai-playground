#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -f .server.pid ]; then
  echo "Wave simulator is not running"
  exit 0
fi
PID=$(cat .server.pid)
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
fi
rm -f .server.pid
echo "Wave simulator stopped"
