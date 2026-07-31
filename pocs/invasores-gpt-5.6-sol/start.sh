#!/usr/bin/env bash
set -e
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$PROJECT_DIR/.gull.pid"
LOG_FILE="$PROJECT_DIR/.gull.log"
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Gull About Town is already running at http://localhost:4173"
  exit 0
fi
cd "$PROJECT_DIR"
nohup "$PROJECT_DIR/node_modules/.bin/vite" --host 0.0.0.0 --port 4173 </dev/null >"$LOG_FILE" 2>&1 &
GAME_PID=$!
echo "$GAME_PID" >"$PID_FILE"
TRIES=0
until curl -fsS http://localhost:4173 >/dev/null 2>&1; do
  if ! kill -0 "$GAME_PID" 2>/dev/null; then
    cat "$LOG_FILE"
    exit 1
  fi
  TRIES=$((TRIES + 1))
  if [ "$TRIES" -ge 30 ]; then
    echo "Game server did not start"
    exit 1
  fi
  sleep 1
done
echo "Gull About Town is running at http://localhost:4173"
