#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require bun

if [ ! -f "$ROOT/packages/engine/dist/index.js" ] || [ ! -f "$ROOT/packages/agents/dist/index.js" ]; then
  log "packages are not built yet"
  "$SCRIPTS/setup.sh"
fi

log "starting"

start_bg arena "$ROOT" bun run --filter '@fly-zord/web' dev
wait_port_up "$(service_port arena)" 60 || fail "arena did not open port $(service_port arena), see $LOGS/arena.log"
log "arena up on $(service_url arena)"

"$SCRIPTS/status.sh"

log "links"
for name in $(service_names); do
  printf "%-14s %s\n" "$name" "$(service_url "$name")"
done
