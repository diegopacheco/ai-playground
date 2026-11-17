#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require bun
PORT="$(service_port ui)"
if port_up "$PORT"; then
  log "ui already up"
else
  UI_PORT="$PORT" start_bg ui "$ROOT" bunx --bun vite --port "$PORT" --strictPort
  wait_port_up "$PORT" 60 || fail "ui did not open port $PORT, see $LOGS/ui.log"
fi
log "ui: $(service_url ui)"
