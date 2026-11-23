#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "starting"

export API_PORT WEB_PORT
API_PORT="$(service_port backend)"
WEB_PORT="$(service_port frontend)"

start_bg backend "$ROOT" bun server/server.ts
wait_port_up "$API_PORT" 60 || fail "backend did not open port $API_PORT, see $LOGS/backend.log"
log "backend up on $(service_url backend)"

start_bg frontend "$ROOT" bun run dev:web
wait_port_up "$WEB_PORT" 60 || fail "frontend did not open port $WEB_PORT, see $LOGS/frontend.log"
log "frontend up on $(service_url frontend)"

"$SCRIPTS/status.sh"

log "links"
printf "%-14s %s\n" backend "$(service_url backend)/api/report"
printf "%-14s %s\n" frontend "$(service_url frontend)"
