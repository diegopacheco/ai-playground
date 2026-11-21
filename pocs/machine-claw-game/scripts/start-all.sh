#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "starting"

require bun
[ -d "$ROOT/node_modules" ] || fail "dependencies are missing, run ./scripts/setup.sh first"

port="$(service_port frontend)"
if port_up "$port" && ! { [ -f "$RUN/frontend.pid" ] && kill -0 "$(cat "$RUN/frontend.pid")" 2>/dev/null; }; then
  fail "port $port is used by another process"
fi

start_bg frontend "$ROOT" bun run dev --host 127.0.0.1 --port "$port" --strictPort
wait_port_up "$port" 60 || fail "frontend did not open port $port, see $LOGS/frontend.log"
log "frontend up on $(service_url frontend)"

"$SCRIPTS/status.sh"

log "links"
for name in $(service_names); do
  printf "%-14s %s\n" "$name" "$(service_url "$name")"
done
