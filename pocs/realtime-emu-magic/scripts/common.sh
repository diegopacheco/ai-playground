#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/ports.env"
APP_PORT="${PORT:-$APP}"
RUN="$ROOT/.run"
mkdir -p "$RUN/logs"
owned_pid() {
  [ -f "$RUN/app.pid" ] || return 1
  APP_PID="$(cat "$RUN/app.pid")"
  case "$APP_PID" in ''|*[!0-9]*) return 1 ;; esac
  kill -0 "$APP_PID" 2>/dev/null || return 1
  ps -p "$APP_PID" -o command= | rg -F -- "$ROOT/server.mjs" >/dev/null
}
healthy() {
  curl --max-time 1 -fsS "http://127.0.0.1:$APP_PORT/health" 2>/dev/null | rg -F '"app":"realtime-emu-magic"' >/dev/null
}
