#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require python3
port="$(service_port landing)"
if port_up "$port"; then
  log "landing already up"
else
  start_bg landing "$ROOT" python3 -m http.server "$port" --bind 127.0.0.1
  wait_port_up "$port" 30 || fail "landing did not open port $port, see $LOGS/landing.log"
fi
log "landing $(service_url landing)/landing/"
