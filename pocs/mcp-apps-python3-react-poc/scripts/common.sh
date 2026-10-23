#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$ROOT/scripts"
RUN="$ROOT/.run"
LOGS="$RUN/logs"
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
  if [ -f "$RUN/$name.pid" ] && kill -0 "$(cat "$RUN/$name.pid")" 2>/dev/null; then
    log "$name already running"
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

log() {
  printf "%s\n" "$*"
}

fail() {
  printf "ERROR: %s\n" "$*" >&2
  exit 1
}

public_host() {
  if [ -f "$SCRIPTS/tailscale.env" ]; then
    grep '^MCP_PUBLIC_HOST=' "$SCRIPTS/tailscale.env" | cut -d= -f2 || true
  fi
}

public_url() {
  if [ "$(service_port funnel)" = "443" ]; then
    printf "https://%s/mcp\n" "$1"
  else
    printf "https://%s:%s/mcp\n" "$1" "$(service_port funnel)"
  fi
}

serve_proxy() {
  tailscale serve status --json 2>/dev/null | python3 -c 'import json,sys;d=json.load(sys.stdin);h=(d.get("Web") or {}).get(sys.argv[1]+":"+sys.argv[2]) or {};print(((h.get("Handlers") or {}).get("/") or {}).get("Proxy",""))' "$1" "$2" 2>/dev/null || true
}

funnel_up() {
  local host
  host="$(public_host)"
  [ -n "$host" ] && command -v tailscale >/dev/null 2>&1 && tailscale funnel status --json 2>/dev/null | grep -q "\"$host:$(service_port funnel)\": true"
}
