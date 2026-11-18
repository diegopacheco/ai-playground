#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$ROOT/scripts"
RUN="$ROOT/.run"
LOGS="$RUN/logs"
cd "$ROOT"

mkdir -p "$RUN" "$LOGS"

SERVICES=""
if [ -f "$SCRIPTS/ports.env" ]; then
  SERVICES="$(grep -v '^[[:space:]]*#' "$SCRIPTS/ports.env" | grep '=' || true)"
fi

service_names() {
  printf "%s\n" "$SERVICES" | sed '/^$/d' | cut -d= -f1
}

service_port() {
  printf "%s\n" "$SERVICES" | sed '/^$/d' | awk -F= -v n="$1" '$1==n{print $2; exit}'
}

service_url() {
  local port
  port="$(service_port "$1")"
  case "$1" in
    postgres) printf "postgresql://localhost:%s\n" "$port" ;;
    mysql|mariadb) printf "mysql://localhost:%s\n" "$port" ;;
    redis|valkey) printf "redis://localhost:%s\n" "$port" ;;
    mongodb) printf "mongodb://localhost:%s\n" "$port" ;;
    *) printf "http://localhost:%s\n" "$port" ;;
  esac
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
  port="$(service_port "$name" || true)"
  if [ -n "$port" ]; then
    wait_port_down "$port" 10 || true
    pid="$(port_pid "$port")"
    if [ -n "$pid" ]; then kill -KILL "$pid" 2>/dev/null || true; fi
  fi
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

POSTGRES_CONTAINER="dbos-poc-fun-py-postgres"
DATABASE_URL="postgresql://dbos:dbos@localhost:$(service_port postgres)/dbos_agent"

agent_env() {
  set -a
  . "$SCRIPTS/agent.conf"
  set +a
  export DBOS_SYSTEM_DATABASE_URL="$DATABASE_URL"
}

wait_http_ok() {
  local tries
  tries="${2:-60}"
  while [ "$tries" -gt 0 ]; do
    if curl -sf -o /dev/null "$1"; then return 0; fi
    sleep 1
    tries=$((tries - 1))
  done
  return 1
}

container_running() {
  [ "$(podman inspect -f '{{.State.Running}}' "$POSTGRES_CONTAINER" 2>/dev/null || true)" = "true" ]
}

wait_postgres() {
  local tries
  tries="${1:-60}"
  while [ "$tries" -gt 0 ]; do
    if podman exec "$POSTGRES_CONTAINER" pg_isready -U dbos -d dbos_agent >/dev/null 2>&1; then return 0; fi
    sleep 1
    tries=$((tries - 1))
  done
  return 1
}

start_agent_service() {
  start_bg agent-service "$ROOT/agent-service" "$ROOT/agent-service/.venv/bin/granian" --interface asgi --host 127.0.0.1 --port "$(service_port agent-service)" --workers 1 agent_service.app:app
  wait_http_ok "$(service_url agent-service)/health" 60 || fail "agent-service did not start, see $LOGS/agent-service.log"
}

start_dashboard() {
  start_bg dashboard-api "$ROOT/dashboard" env DATABASE_URL="$DATABASE_URL" PORT="$(service_port dashboard-api)" bun server/main.ts
  wait_http_ok "$(service_url dashboard-api)/api/health" 30 || fail "dashboard-api did not start, see $LOGS/dashboard-api.log"
  start_bg dashboard "$ROOT/dashboard" bun run dev --port "$(service_port dashboard)" --strictPort
  wait_port_up "$(service_port dashboard)" 30 || fail "dashboard did not start, see $LOGS/dashboard.log"
}

kill_agent_service() {
  local pid
  pid="$(cat "$RUN/agent-service.pid")"
  pkill -KILL -P "$pid" 2>/dev/null || true
  kill -KILL "$pid" 2>/dev/null || true
  rm -f "$RUN/agent-service.pid"
  wait_port_down "$(service_port agent-service)" 10 || true
}
