#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require node
[ -d "$ROOT/node_modules/leaflet" ] || fail "dependencies missing, run scripts/setup.sh"

port="$(service_port WEB)"
if port_up "$port"; then
  log "WEB already up on port $port"
else
  PORT="$port" start_bg WEB "$ROOT" node server/server.mjs
  wait_port_up "$port" 60 || fail "WEB did not open port $port, see $LOGS/WEB.log"
fi

"$SCRIPTS/status.sh"

log "links"
for name in $(service_names); do
  printf "%-14s %s\n" "$name" "$(service_url "$name")"
done
