#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

agent_env
ADMIN="$(service_url restate-admin)"
podman-compose -f "$ROOT/podman-compose.yml" up -d >"$LOGS/podman-compose.log" 2>&1 || fail "podman-compose up failed, see $LOGS/podman-compose.log"
wait_http_ok "$ADMIN/health" 60 || fail "restate did not become healthy"
log "restate up"

start_bg agent-service "$ROOT/agent-service" bun run src/main.ts
wait_port_up "$(service_port agent-service)" 30 || fail "agent-service did not start, see $LOGS/agent-service.log"

curl -sf -X POST "$ADMIN/deployments" -H 'content-type: application/json' \
  -d "{\"uri\":\"http://host.containers.internal:$(service_port agent-service)\",\"force\":true}" >"$LOGS/register.log" \
  || fail "registering agent-service in restate failed"
log "agent-service registered in restate"

start_bg ui "$ROOT/ui" bun run dev --port "$(service_port ui)" --strictPort
wait_port_up "$(service_port ui)" 30 || fail "ui did not start, see $LOGS/ui.log"

log ""
log "ui              $(service_url ui)"
log "restate ui      $(service_url restate-admin)/ui"
log "restate ingress $(service_url restate-ingress)"
log "restate admin   $(service_url restate-admin)"
log "agent-service   $(service_url agent-service)"
