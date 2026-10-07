#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "starting"

require node
port="$(service_port younap)"
start_bg younap "$ROOT" env PORT="$port" node server/server.js
wait_port_up "$port" 60 || fail "younap did not open port $port, see $LOGS/younap.log"
log "younap up on $(service_url younap)"

"$SCRIPTS/status.sh"

log "links"
for name in $(service_names); do
  printf "%-14s %s\n" "$name" "$(service_url "$name")"
done
