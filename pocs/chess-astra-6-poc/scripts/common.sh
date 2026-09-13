#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/ports.env"
RUN="$ROOT/.run"
mkdir -p "$RUN/logs"
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
require() { command -v "$1" >/dev/null 2>&1 || fail "$1 is required"; }
port_pid() { lsof -tiTCP:"$FRONTEND" -sTCP:LISTEN 2>/dev/null | head -1 || true; }
project_vite() {
  local command cwd
  command="$(ps -p "$1" -o command= 2>/dev/null)" || return 1
  cwd="$(lsof -a -p "$1" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p')"
  case "$command" in *vite*) [ "$cwd" = "$ROOT" ] ;; *) return 1 ;; esac
}
owned_pid() {
  [ -f "$RUN/frontend.pid" ] || return 1
  local pid
  pid="$(cat "$RUN/frontend.pid")"
  kill -0 "$pid" 2>/dev/null || return 1
  project_vite "$pid" || return 1
  printf '%s\n' "$pid"
}
