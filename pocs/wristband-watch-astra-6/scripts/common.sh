#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/ports.env"
RUN="$ROOT/.run"
mkdir -p "$RUN/logs"
URL="http://127.0.0.1:$FRONTEND"
fail() {
  printf '%s\n' "$*" >&2
  exit 1
}
owned_pid() {
  [ -f "$RUN/frontend.pid" ] || return 1
  local pid
  pid="$(cat "$RUN/frontend.pid")"
  case "$pid" in ''|*[!0-9]*) return 1 ;; esac
  kill -0 "$pid" 2>/dev/null || return 1
  ps -p "$pid" -o command= | rg -F -- "$ROOT/server.mjs" >/dev/null || return 1
  printf '%s' "$pid"
}
healthy() {
  curl --fail --silent --max-time 1 "$URL" | rg -q '<title>ASTRA / ONE'
}
