#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "starting"
require bun

port="$(service_port frontend)"
[ -n "$port" ] || fail "frontend port missing in scripts/ports.env"

if port_up "$port"; then
  log "frontend already listening on $port"
else
  start_bg frontend "$ROOT" env "FRONTEND_PORT=$port" bun run dev -- --port "$port" --host 127.0.0.1
  wait_port_up "$port" 60 || fail "frontend did not open port $port, see $LOGS/frontend.log"
fi

log "frontend up on $(service_url frontend)"
"$SCRIPTS/status.sh"

log "links"
for name in $(service_names); do
  printf "%-14s %s\n" "$name" "$(service_url "$name")"
done
