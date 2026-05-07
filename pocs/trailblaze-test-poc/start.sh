#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PORT=8080
PID_FILE=".server.pid"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Server already running on port $PORT (pid $(cat "$PID_FILE"))"
  exit 0
fi

python3 -m http.server "$PORT" --directory sample-app >/tmp/trailblaze-poc-server.log 2>&1 &
echo $! > "$PID_FILE"

ATTEMPTS=0
until curl -fsS "http://localhost:$PORT/" >/dev/null 2>&1; do
  ATTEMPTS=$((ATTEMPTS + 1))
  if [ "$ATTEMPTS" -gt 30 ]; then
    echo "Server failed to start"
    cat /tmp/trailblaze-poc-server.log
    exit 1
  fi
  sleep 1
done

echo "Server up at http://localhost:$PORT/ (pid $(cat "$PID_FILE"))"
