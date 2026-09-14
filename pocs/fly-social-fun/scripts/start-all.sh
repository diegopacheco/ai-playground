#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

port="$(service_port buzzr)"
[ -n "$port" ] || fail "buzzr is not declared in scripts/ports.env"
[ -d "$ROOT/node_modules/three" ] || fail "dependencies missing, run ./scripts/setup.sh first"

log "starting"

if port_up "$port" && ! { [ -f "$RUN/buzzr.pid" ] && kill -0 "$(cat "$RUN/buzzr.pid")" 2>/dev/null; }; then
  fail "port $port is taken by pid $(port_pid "$port"), stop it or change scripts/ports.env"
fi

start_bg buzzr "$ROOT" env PORT="$port" node server/server.js
wait_port_up "$port" 60 || fail "buzzr did not open port $port, see $LOGS/buzzr.log"
log "buzzr up on $port"

"$SCRIPTS/status.sh"
