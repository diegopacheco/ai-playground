#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "stopping"

stop_bg frontend
stop_bg backend

require podman-compose
podman-compose down

for name in $(owned_service_names); do
  port="$(service_port "$name")"
  wait_port_down "$port" 15 || fail "$name still listening on $port"
done

log "stopped"
