#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "starting"
port_up "$(service_port registry)" || fail "registry is down, run scripts/setup.sh"
[ -x "$BIN/console" ] || "$SCRIPTS/build-backend.sh"
CONSOLE_PORT="$(service_port console)"
FRONTEND_PORT="$(service_port frontend)"

start_bg console "$ROOT" env PORT="$CONSOLE_PORT" ATE_BIN="$BIN/kubectl-ate" KUBE_CONTEXT="$KCTX" ATESPACE="$ATESPACE" TEMPLATE="$TEMPLATE" "$BIN/console"
wait_port_up "$CONSOLE_PORT" 60 || fail "console did not open port $CONSOLE_PORT, see $LOGS/console.log"
log "console up on $(service_url console)/api"

start_bg frontend "$ROOT/frontend" env API_PORT="$CONSOLE_PORT" UI_PORT="$FRONTEND_PORT" bunx vite --host 127.0.0.1
wait_port_up "$FRONTEND_PORT" 60 || fail "frontend did not open port $FRONTEND_PORT, see $LOGS/frontend.log"
log "frontend up on $(service_url frontend)"

"$SCRIPTS/status.sh"

log "links"
printf "%-14s %s\n" console "$(service_url console)/api/actors"
printf "%-14s %s\n" frontend "$(service_url frontend)"
printf "%-14s %s\n" registry "$(service_url registry)/v2/_catalog"
