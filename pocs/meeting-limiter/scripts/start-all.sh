#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "starting"

require go
require bun

api_port="$(service_port api)"
ui_port="$(service_port ui)"

if ! port_up "$api_port"; then
  ( cd "$ROOT/backend" && go build -o "$RUN/bin/api" ./cmd/server ) || fail "backend build failed"
  mkdir -p "$ROOT/data"
  start_bg api "$ROOT/backend" env PORT="$api_port" DATA_FILE="$ROOT/data/meetings.json" "$RUN/bin/api"
fi
wait_port_up "$api_port" 60 || fail "api did not open port $api_port, see $LOGS/api.log"
log "api up on $(service_url api)"

if ! port_up "$ui_port"; then
  start_bg ui "$ROOT/frontend" env API_PORT="$api_port" UI_PORT="$ui_port" bun run dev
fi
wait_port_up "$ui_port" 60 || fail "ui did not open port $ui_port, see $LOGS/ui.log"
log "ui up on $(service_url ui)"

"$SCRIPTS/status.sh"

log "links"
for name in $(service_names); do
  printf "%-14s %s\n" "$name" "$(service_url "$name")"
done
