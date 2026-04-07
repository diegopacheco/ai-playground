#!/bin/zsh
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$ROOT_DIR/.server.pid"
LOG_FILE="$ROOT_DIR/.server.log"
PORT=8091
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "server already running on http://127.0.0.1:$PORT"
  exit 0
fi
rm -f "$PID_FILE"
cd "$ROOT_DIR"
python3 -m http.server "$PORT" --bind 127.0.0.1 >"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
for _ in 1 2 3 4 5; do
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "running on http://127.0.0.1:$PORT"
    exit 0
  fi
  sleep 1
done
rm -f "$PID_FILE"
echo "server failed to start"
exit 1
