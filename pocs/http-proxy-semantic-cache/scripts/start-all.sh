#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "starting"

require_ollama_model
start_redis

[ -x "$ROOT/backend/target/release/semantic-cache-proxy" ] || fail "backend is not built, run ./scripts/setup.sh"
start_bg backend "$ROOT/backend" ./target/release/semantic-cache-proxy
wait_port_up "$PROXY_PORT" 60 || fail "backend did not open port $PROXY_PORT, see $LOGS/backend.log"
log "backend up on $(service_url backend)"

start_bg ui "$ROOT/ui" bun run dev
wait_port_up "$UI_PORT" 60 || fail "ui did not open port $UI_PORT, see $LOGS/ui.log"
log "ui up on $(service_url ui)"

"$SCRIPTS/status.sh"

log "links"
for name in $(service_names); do
  printf "%-10s %s\n" "$name" "$(service_url "$name")"
done
