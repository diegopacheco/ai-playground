#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

agent_env
podman-compose -f "$ROOT/podman-compose.yml" up -d >"$LOGS/podman-compose.log" 2>&1 || fail "podman-compose up failed, see $LOGS/podman-compose.log"
wait_http_ok "$TEMPORAL_UI_URL" 90 || fail "temporal did not become ready, see podman logs $TEMPORAL_CONTAINER"
wait_port_up "$(service_port temporal)" 30 || fail "temporal frontend is not listening"
log "temporal up"

start_worker
log "worker up, polling the agent-tasks queue"
start_agent_api
log "agent-api up"

start_bg ui "$ROOT/ui" bun run dev --port "$(service_port ui)" --strictPort
wait_port_up "$(service_port ui)" 30 || fail "ui did not start, see $LOGS/ui.log"

log ""
log "ui              $(service_url ui)"
log "temporal ui     $TEMPORAL_UI_URL"
log "temporal        $TEMPORAL_ADDRESS"
log "agent-api       $(service_url agent-api)"
