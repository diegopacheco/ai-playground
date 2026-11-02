#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$ROOT/scripts"
RUN="$ROOT/.run"
LOGS="$RUN/logs"
MCP_NAME="mcp-events"
cd "$ROOT"

mkdir -p "$RUN" "$LOGS"

SERVICES="$(grep -v '^[[:space:]]*#' "$SCRIPTS/ports.env" | grep '=' || true)"

service_names() {
  printf "%s\n" "$SERVICES" | sed '/^$/d' | cut -d= -f1
}

service_port() {
  printf "%s\n" "$SERVICES" | sed '/^$/d' | awk -F= -v n="$1" '$1==n{print $2; exit}'
}

service_url() {
  printf "http://localhost:%s%s\n" "$(service_port "$1")" "${2:-}"
}

mcp_url() {
  service_url backend /mcp
}

port_pid() {
  lsof -ti "tcp:$1" -sTCP:LISTEN 2>/dev/null | head -1 || true
}

port_up() {
  [ -n "$(port_pid "$1")" ]
}

wait_port_up() {
  local tries
  tries="${2:-60}"
  while [ "$tries" -gt 0 ]; do
    if port_up "$1"; then return 0; fi
    sleep 1
    tries=$((tries - 1))
  done
  return 1
}

wait_port_down() {
  local tries
  tries="${2:-30}"
  while [ "$tries" -gt 0 ]; do
    if ! port_up "$1"; then return 0; fi
    sleep 1
    tries=$((tries - 1))
  done
  return 1
}

start_bg() {
  local name dir
  name="$1"
  dir="$2"
  shift 2
  if port_up "$(service_port "$name")"; then
    log "$name already running on port $(service_port "$name")"
    return 0
  fi
  ( cd "$dir" && exec "$@" >"$LOGS/$name.log" 2>&1 ) &
  echo $! >"$RUN/$name.pid"
  log "$name started pid $!"
}

stop_bg() {
  local name pid port
  name="$1"
  if [ -f "$RUN/$name.pid" ]; then
    pid="$(cat "$RUN/$name.pid")"
    if kill -0 "$pid" 2>/dev/null; then
      pkill -TERM -P "$pid" 2>/dev/null || true
      kill -TERM "$pid" 2>/dev/null || true
    fi
    rm -f "$RUN/$name.pid"
  fi
  port="$(service_port "$name")"
  wait_port_down "$port" 10 || true
  pid="$(port_pid "$port")"
  if [ -n "$pid" ]; then kill -KILL "$pid" 2>/dev/null || true; fi
  wait_port_down "$port" 10 || fail "$name still holds port $port"
  log "$name stopped"
}

open_url() {
  if command -v open >/dev/null 2>&1; then
    open "$1"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$1"
  else
    fail "no browser opener found, open $1 manually"
  fi
}

require() {
  command -v "$1" >/dev/null 2>&1 || fail "$1 is required but not installed"
}

confirm() {
  local answer
  printf "%s [y/N] " "$1"
  read -r answer || answer=""
  [ "$answer" = "y" ] || [ "$answer" = "Y" ]
}

pick_targets() {
  local verb
  verb="$1"
  shift
  if [ "$#" -gt 0 ]; then
    printf "%s\n" "$@"
    return 0
  fi
  if confirm "$verb Claude Code?" >&2; then echo claude; fi
  if confirm "$verb Codex?" >&2; then echo codex; fi
}

log() {
  printf "%s\n" "$*"
}

fail() {
  printf "ERROR: %s\n" "$*" >&2
  exit 1
}
