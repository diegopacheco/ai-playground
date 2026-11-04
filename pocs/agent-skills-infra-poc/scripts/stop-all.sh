#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "stopping"
stop_bg ui
for name in $API_SERVICES; do
  stop_bg "$name"
done

compose_env
podman-compose down >"$LOGS/compose-down.log" 2>&1 || fail "podman-compose down failed, see $LOGS/compose-down.log"

for name in $(service_names); do
  port="$(service_port "$name")"
  wait_port_down "$port" 15 || fail "$name still listening on $port"
done

log "stopped"
