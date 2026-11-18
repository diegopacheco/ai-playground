#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

stop_bg ui
stop_bg agent-api
stop_tree worker
podman-compose -f "$ROOT/podman-compose.yml" down >"$LOGS/podman-compose.log" 2>&1 || fail "podman-compose down failed, see $LOGS/podman-compose.log"
log "temporal stopped"
