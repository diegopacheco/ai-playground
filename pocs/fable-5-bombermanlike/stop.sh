#!/bin/bash
cd "$(dirname "$0")"
if [ -f server.pid ]; then
  PID=$(cat server.pid)
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    echo "BLASTGRID stopped"
  else
    echo "server not running"
  fi
  rm -f server.pid
else
  echo "server not running"
fi
