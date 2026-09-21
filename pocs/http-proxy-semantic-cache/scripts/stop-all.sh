#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "stopping"

stop_bg ui
stop_bg backend

require podman-compose
export_ports
podman-compose down >"$LOGS/redis.log" 2>&1 || fail "podman-compose down failed, see $LOGS/redis.log"
wait_port_down "$REDIS_PORT" 30 || true

for name in $(service_names); do
  port="$(service_port "$name")"
  if port_up "$port"; then
    fail "$name still listening on $port"
  fi
done

log "stopped"
