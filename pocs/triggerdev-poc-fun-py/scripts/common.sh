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

WEBAPP_CONTAINER="triggerdev-poc-webapp"
POSTGRES_CONTAINER="triggerdev-poc-postgres"
STACK_CONF="$RUN/stack.conf"
TRIGGER_CONF="$RUN/trigger.conf"

ensure_stack_conf() {
  [ -f "$STACK_CONF" ] && return 0
  {
    for name in SESSION_SECRET MAGIC_LINK_SECRET ENCRYPTION_KEY PROVIDER_SECRET COORDINATOR_SECRET MANAGED_WORKER_SECRET POSTGRES_PASSWORD CLICKHOUSE_PASSWORD OBJECT_STORE_SECRET_ACCESS_KEY; do
      printf "%s=%s\n" "$name" "$(openssl rand -hex 16)"
    done
  } >"$STACK_CONF"
  chmod 600 "$STACK_CONF"
  log "generated local secrets in $STACK_CONF"
}

compose() {
  podman-compose --env-file "$STACK_CONF" -f "$ROOT/podman-compose.yml" "$@"
}

load_conf() {
  [ -f "$1" ] || return 0
  set -a
  . "$1"
  set +a
}

agent_env() {
  load_conf "$SCRIPTS/agent.conf"
  load_conf "$TRIGGER_CONF"
  export TRIGGER_API_URL="$(service_url trigger-webapp)"
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

wait_log() {
  local tries
  tries="${3:-60}"
  while [ "$tries" -gt 0 ]; do
    if grep -q "$2" "$1" 2>/dev/null; then return 0; fi
    sleep 1
    tries=$((tries - 1))
  done
  return 1
}

container_running() {
  [ "$(podman inspect -f '{{.State.Running}}' "$WEBAPP_CONTAINER" 2>/dev/null || true)" = "true" ]
}

pid_running() {
  [ -f "$RUN/$1.pid" ] && kill -0 "$(cat "$RUN/$1.pid")" 2>/dev/null
}

kill_tree() {
  local child
  for child in $(pgrep -P "$1" 2>/dev/null || true); do
    kill_tree "$child" "$2"
  done
  kill "-$2" "$1" 2>/dev/null || true
}

stop_tree() {
  if [ -f "$RUN/$1.pid" ]; then
    kill_tree "$(cat "$RUN/$1.pid")" "${2:-TERM}"
    rm -f "$RUN/$1.pid"
  fi
  log "$1 stopped"
}

start_trigger_dev() {
  rm -f "$LOGS/trigger-dev.log"
  start_bg trigger-dev "$ROOT/tasks" ./node_modules/.bin/trigger dev start --skip-update-check --skip-telemetry --api-url "$TRIGGER_API_URL"
  wait_log "$LOGS/trigger-dev.log" "Local worker ready" 120 || fail "trigger dev did not register the tasks, see $LOGS/trigger-dev.log"
  log "trigger dev worker ready, tasks registered"
}

start_agent_api() {
  start_bg agent-api "$ROOT/api" "$ROOT/api/.venv/bin/granian" --interface asgi --host 127.0.0.1 --port "$(service_port agent-api)" --workers 1 agent_api.app:app
  wait_http_ok "$(service_url agent-api)/health" 60 || fail "agent-api did not start, see $LOGS/agent-api.log"
}
