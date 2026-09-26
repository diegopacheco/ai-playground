#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require python3
port="$(service_port web)"
if port_up "$port" && [ ! -f "$RUN/web.pid" ]; then
  fail "port $port is taken by another process"
fi
start_bg web "$ROOT" python3 -m http.server "$port" --bind 127.0.0.1
wait_port_up "$port" 60 || fail "web did not open port $port, see $LOGS/web.log"

"$SCRIPTS/status.sh"
log "links"
printf "%-14s %s\n" web "$(service_url web)"
