#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
if pid="$(owned_pid)"; then
  healthy || fail "Process $pid is running but the UI is not healthy"
  printf 'Frontend already running on %s, pid %s\n' "$FRONTEND" "$pid"
  exit 0
fi
if lsof -ti "tcp:$FRONTEND" -sTCP:LISTEN >/dev/null 2>&1; then
  fail "Port $FRONTEND is already in use"
fi
nohup node "$ROOT/server.mjs" > "$RUN/logs/frontend.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$RUN/frontend.pid"
for attempt in $(seq 1 30); do
  if healthy; then
    printf 'Frontend UP %s pid %s\n' "$URL" "$pid"
    exit 0
  fi
  kill -0 "$pid" 2>/dev/null || fail "Frontend exited; read .run/logs/frontend.log"
  sleep 1
done
kill "$pid" 2>/dev/null || true
rm -f "$RUN/frontend.pid"
fail 'Frontend did not become healthy within 30 checks'
