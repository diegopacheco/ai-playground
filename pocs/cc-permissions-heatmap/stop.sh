#!/usr/bin/env bash
cd "$(dirname "$0")"

if [ -f .server.pid ]; then
  PID="$(cat .server.pid)"
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null
    for i in $(seq 1 10); do
      if kill -0 "$PID" 2>/dev/null; then sleep 1; else break; fi
    done
    echo "Stopped server (pid $PID)"
  else
    echo "No running server for pid $PID"
  fi
  rm -f .server.pid
else
  echo "No .server.pid found"
fi
