#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require node
[ -f "$ROOT/data/movies.json" ] || fail "data/movies.json is missing, run scripts/setup.sh"
[ -d "$ROOT/node_modules/leaflet" ] || fail "node_modules is missing, run scripts/setup.sh"
port="$(service_port api)"
if port_up "$port"; then
  log "api already listening on $port"
else
  start_bg api "$ROOT" env API_PORT="$port" node server/server.mjs
  wait_port_up "$port" 30 || fail "api did not start, see .run/logs/api.log"
fi
log "api  $(service_url api)/api/health"
log "ui   $(service_url api)"
