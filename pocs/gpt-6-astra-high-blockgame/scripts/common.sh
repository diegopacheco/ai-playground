#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/ports.env"
RUN="$ROOT/.run"
mkdir -p "$RUN/logs"
fail() { printf '%s\n' "$*" >&2; exit 1; }
owned() {
  [ -f "$RUN/web.pid" ] || return 1
  local pid
  pid="$(cat "$RUN/web.pid")"
  case "$pid" in ''|*[!0-9]*) return 1 ;; esac
  kill -0 "$pid" 2>/dev/null && ps -p "$pid" -o command= | grep -F "$ROOT/server.mjs" >/dev/null
}
healthy() { curl --fail --silent --max-time 1 "http://localhost:$WEB/health" | grep -q '"app":"brisa"'; }
