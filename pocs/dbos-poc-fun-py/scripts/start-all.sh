#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

agent_env
podman-compose -f "$ROOT/podman-compose.yml" up -d >"$LOGS/podman-compose.log" 2>&1 || fail "podman-compose up failed, see $LOGS/podman-compose.log"
wait_postgres 60 || fail "postgres did not become ready"
log "postgres up"

start_agent_service
log "agent-service up, DBOS launched and recovered any pending workflow"

start_bg ui "$ROOT/ui" bun run dev --port "$(service_port ui)" --strictPort
wait_port_up "$(service_port ui)" 30 || fail "ui did not start, see $LOGS/ui.log"

log ""
log "ui              $(service_url ui)"
log "agent-service   $(service_url agent-service)"
log "postgres        $DATABASE_URL"
