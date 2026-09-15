#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "starting"

port="$(service_port game)"
if port_up "$port"; then
  log "game already running"
else
  require python3
  build_game
  start_bg game "$ROOT/public" python3 -m http.server "$port" --bind 127.0.0.1
  wait_port_up "$port" 60 || fail "game did not open port $port, see $LOGS/game.log"
fi
log "game up on $(service_url game)"

"$SCRIPTS/status.sh"

log "links"
for name in $(service_names); do
  printf "%-14s %s\n" "$name" "$(service_url "$name")"
done
