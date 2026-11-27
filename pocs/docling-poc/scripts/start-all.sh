#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "starting"
BACKEND_PORT="$(service_port backend)"
FRONTEND_PORT="$(service_port frontend)"

start_bg backend "$ROOT/backend" uv run granian --interface asgi --host 127.0.0.1 --port "$BACKEND_PORT" app.main:app
wait_port_up "$BACKEND_PORT" 60 || fail "backend did not open port $BACKEND_PORT, see $LOGS/backend.log"
log "backend up on $(service_url backend)/api"

start_bg frontend "$ROOT/frontend" env API_PORT="$BACKEND_PORT" UI_PORT="$FRONTEND_PORT" bunx vite --host 127.0.0.1
wait_port_up "$FRONTEND_PORT" 60 || fail "frontend did not open port $FRONTEND_PORT, see $LOGS/frontend.log"
log "frontend up on $(service_url frontend)"

"$SCRIPTS/status.sh"

log "links"
printf "%-14s %s\n" backend "$(service_url backend)/api/health"
printf "%-14s %s\n" frontend "$(service_url frontend)"
