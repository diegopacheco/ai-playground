#!/usr/bin/env bash
set -e
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$PROJECT_DIR/.gull.pid"
if [ ! -f "$PID_FILE" ]; then
  echo "Gull About Town is not running"
  exit 0
fi
GAME_PID="$(cat "$PID_FILE")"
if kill -0 "$GAME_PID" 2>/dev/null; then
  kill "$GAME_PID"
fi
rm "$PID_FILE"
echo "Gull About Town stopped"
