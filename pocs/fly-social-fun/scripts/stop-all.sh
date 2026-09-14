#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "stopping"

stop_bg buzzr

port="$(service_port buzzr)"
if port_up "$port"; then
  fail "buzzr still listening on $port"
fi

log "stopped"
