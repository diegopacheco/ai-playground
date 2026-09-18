#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "stopping"
stop_bg app

port="$(service_port app)"
if port_up "$port"; then
  fail "app still listening on $port"
fi

log "stopped"
