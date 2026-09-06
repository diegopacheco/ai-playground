#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/ports.env"
RUN="$ROOT/.run"
mkdir -p "$RUN/logs"
port_pid() { lsof -ti "tcp:$WEB" -sTCP:LISTEN 2>/dev/null | head -1 || true; }
fail() { printf '%s\n' "$*" >&2; exit 1; }
owned_pid() {
  [ -f "$RUN/web.pid" ] || return 1
  local pid command
  pid="$(cat "$RUN/web.pid")"
  command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  case "$command" in
    *"node $ROOT/server.mjs"*) printf '%s' "$pid" ;;
    *) return 1 ;;
  esac
}
