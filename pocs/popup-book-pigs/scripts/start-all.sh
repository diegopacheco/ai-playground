#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "starting"
require bun
[ -d "$ROOT/node_modules" ] || fail "dependencies missing, run ./scripts/setup.sh first"
port="$(service_port frontend)"
if port_up "$port"; then
  log "frontend already running"
else
  start_bg frontend "$ROOT" env PORT="$port" bunx vite --port "$port" --strictPort
  wait_port_up "$port" 60 || fail "frontend did not open port $port, see $LOGS/frontend.log"
fi
"$SCRIPTS/status.sh"
log "links"
for name in $(service_names); do
  printf "%-14s %s\n" "$name" "$(service_url "$name")"
done
