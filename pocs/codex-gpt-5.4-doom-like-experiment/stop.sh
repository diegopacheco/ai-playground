#!/bin/zsh
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$ROOT_DIR/.server.pid"
LOG_FILE="$ROOT_DIR/.server.log"
if [ ! -f "$PID_FILE" ]; then
  echo "server is not running"
  exit 0
fi
PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  for _ in 1 2 3 4 5; do
    if ! kill -0 "$PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done
fi
rm -f "$PID_FILE"
rm -f "$LOG_FILE"
echo "server stopped"
