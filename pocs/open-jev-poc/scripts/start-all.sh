#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

[ -x "$PY" ] || fail "run ./scripts/setup.sh first"

log "starting"
port="$(service_port app)"
if port_up "$port"; then
  log "app already up"
else
  start_bg app "$ROOT" env PORT="$port" "$PY" -m semif_poc.server
  wait_port_up "$port" 60 || fail "app did not open port $port, see .run/logs/app.log"
fi

"$SCRIPTS/status.sh"

log "links"
printf "%-14s %s\n" app "$(service_url app)"
printf "%-14s %s\n" api "$(service_url app)/api/decide"
