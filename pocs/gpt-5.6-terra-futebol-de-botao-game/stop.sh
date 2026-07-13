#!/bin/bash
set -e
if [ -f .server.pid ]; then
  PID=$(cat .server.pid)
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
  fi
  rm -f .server.pid
fi
echo "Mesa 12 stopped"
