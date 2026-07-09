#!/usr/bin/env bash
set -euo pipefail
PORT="${PORT:-8097}"
PID_FILE=".server.pid"
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Server already running at http://127.0.0.1:$PORT"
  exit 0
fi
LOG_FILE="/tmp/rubik-cube-server.log"
: > "$LOG_FILE"
nohup python3 -m http.server "$PORT" --bind 127.0.0.1 >"$LOG_FILE" 2>&1 &
echo "$!" > "$PID_FILE"
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    cat "$LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
  fi
  if python3 - "$PORT" <<'PY' >/dev/null 2>&1
import sys
import urllib.request
port = sys.argv[1]
body = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1).read().decode()
raise SystemExit(0 if "Rubik's Cube" in body else 1)
PY
  then
    echo "Server running at http://127.0.0.1:$PORT"
    exit 0
  fi
  sleep 1
done
kill "$(cat "$PID_FILE")" 2>/dev/null || true
rm -f "$PID_FILE"
echo "Server did not start"
exit 1
