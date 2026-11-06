#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

stop_bg gateway
stop_bg text
stop_bg math

for name in $(service_names); do
  if port_up "$(service_port "$name")"; then
    fail "$name still listening on $(service_port "$name")"
  fi
done

log "stopped"
