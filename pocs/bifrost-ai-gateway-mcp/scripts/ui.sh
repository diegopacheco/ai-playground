#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for name in app bifrost; do
  port="$(service_port "$name")"
  port_up "$port" || fail "$name is not running on $port, run ./scripts/start-all.sh first"
done

log "opening $(service_url app)"
open_url "$(service_url app)"
log "opening $(service_url bifrost)"
open_url "$(service_url bifrost)"
