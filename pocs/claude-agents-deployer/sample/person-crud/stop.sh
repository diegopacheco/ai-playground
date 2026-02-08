#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/.pids"

if [ ! -f "$PID_FILE" ]; then
  echo "No PID file found"
  exit 1
fi

while read -r PID; do
  if kill -0 "$PID" 2>/dev/null; then
    pgrep -P "$PID" 2>/dev/null | while read -r CHILD; do
      pgrep -P "$CHILD" 2>/dev/null | xargs kill 2>/dev/null
      kill "$CHILD" 2>/dev/null
    done
    kill "$PID" 2>/dev/null
    echo "Killed process $PID and children"
  else
    echo "Process $PID not running"
  fi
done < "$PID_FILE"

lsof -ti :8080 2>/dev/null | xargs kill 2>/dev/null
lsof -ti :3000 2>/dev/null | xargs kill 2>/dev/null

rm -f "$PID_FILE"
rm -f "$SCRIPT_DIR/backend.log" "$SCRIPT_DIR/frontend.log"
echo "Cleanup done"
