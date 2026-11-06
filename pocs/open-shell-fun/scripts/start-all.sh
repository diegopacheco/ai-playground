#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require openshell
require bun

GATEWAY_PORT="$(service_port gateway)"
BACKEND_PORT="$(service_port backend)"
FRONTEND_PORT="$(service_port frontend)"

if ! port_up "$GATEWAY_PORT"; then
  log "starting OpenShell gateway service"
  brew services start nvidia/openshell/openshell >/dev/null
fi
wait_port_up "$GATEWAY_PORT" 60 || fail "gateway did not listen on $GATEWAY_PORT"
openshell status >/dev/null 2>&1 || fail "openshell status cannot reach the gateway"

if ! port_up "$BACKEND_PORT"; then
  start_bg backend "$ROOT/backend" env BACKEND_PORT="$BACKEND_PORT" SANDBOX_IMAGE="$SANDBOX_IMAGE" bun run src/server.ts
fi
wait_port_up "$BACKEND_PORT" 60 || fail "backend did not listen on $BACKEND_PORT, see $LOGS/backend.log"

if ! port_up "$FRONTEND_PORT"; then
  start_bg frontend "$ROOT/frontend" env BACKEND_PORT="$BACKEND_PORT" FRONTEND_PORT="$FRONTEND_PORT" ./node_modules/.bin/vite
fi
wait_port_up "$FRONTEND_PORT" 60 || fail "frontend did not listen on $FRONTEND_PORT, see $LOGS/frontend.log"

log "gateway  $(service_url gateway)"
log "backend  $(service_url backend)/api"
log "frontend $(service_url frontend)"
