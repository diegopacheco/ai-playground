#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
if ! owned_pid; then
  stale_pids="$(lsof -ti "tcp:$APP_PORT" -sTCP:LISTEN 2>/dev/null || true)"
  for stale_pid in $stale_pids; do
    stale_command="$(ps -p "$stale_pid" -o command= 2>/dev/null || true)"
    stale_cwd="$(lsof -p "$stale_pid" -a -d cwd -Fn 2>/dev/null | sed -n 's/^n//p')"
    case "$stale_command:$stale_cwd" in
      *"$ROOT/server.mjs"*|*":$ROOT")
        kill "$stale_pid" 2>/dev/null || true
        for attempt in $(seq 1 30); do
          if ! lsof -ti "tcp:$APP_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
            rm -f "$RUN/app.pid"
            printf 'App stopped.\n'
            exit 0
          fi
          sleep 1
        done
        printf 'App is still shutting down, pid %s.\n' "$stale_pid" >&2
        exit 1
        ;;
    esac
  done
  rm -f "$RUN/app.pid"
  printf 'App is already stopped.\n'
  exit 0
fi
kill "$APP_PID"
for attempt in $(seq 1 30); do
  if ! kill -0 "$APP_PID" 2>/dev/null; then
    rm -f "$RUN/app.pid"
    printf 'App stopped.\n'
    exit 0
  fi
  sleep 1
done
printf 'App is still shutting down, pid %s.\n' "$APP_PID" >&2
exit 1
