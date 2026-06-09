#!/usr/bin/env bash
cd "$(dirname "$0")"

if [ -f .server.pid ]; then
  PID=$(cat .server.pid)
  kill "$PID" 2>/dev/null || true
  rm -f .server.pid
  echo "Stopped server pid $PID"
fi

pkill -f "next dev -p 3434" 2>/dev/null || true
echo "Mind Reader stopped"
