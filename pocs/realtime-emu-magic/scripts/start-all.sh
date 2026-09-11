#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
if owned_pid; then
  printf 'App already running, pid %s.\n' "$APP_PID"
  exit 0
fi
if lsof -ti "tcp:$APP_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  listener_pids="$(lsof -ti "tcp:$APP_PORT" -sTCP:LISTEN 2>/dev/null || true)"
  for listener_pid in $listener_pids; do
    listener_command="$(ps -p "$listener_pid" -o command= 2>/dev/null || true)"
    printf 'Port %s is already occupied by pid %s%s. Run ./scripts/stop-all.sh.\n' "$APP_PORT" "$listener_pid" "${listener_command:+ ($listener_command)}" >&2
  done
  exit 1
fi
nohup env PORT="$APP_PORT" node --env-file-if-exists="$ROOT/.env.local" "$ROOT/server.mjs" >"$RUN/logs/app.log" 2>&1 &
APP_PID=$!
printf '%s\n' "$APP_PID" >"$RUN/app.pid"
for attempt in $(seq 1 30); do
  if ! kill -0 "$APP_PID" 2>/dev/null; then
    cat "$RUN/logs/app.log" >&2
    rm -f "$RUN/app.pid"
    exit 1
  fi
  if healthy; then
    printf 'App running at http://127.0.0.1:%s\n' "$APP_PORT"
    exit 0
  fi
  sleep 1
done
kill "$APP_PID" 2>/dev/null || true
rm -f "$RUN/app.pid"
printf 'App did not become healthy. Check .run/logs/app.log.\n' >&2
exit 1
