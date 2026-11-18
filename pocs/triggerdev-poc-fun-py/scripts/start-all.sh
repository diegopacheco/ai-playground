#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ensure_stack_conf
agent_env
compose up -d >"$LOGS/podman-compose.log" 2>&1 || fail "podman-compose up failed, see $LOGS/podman-compose.log"
wait_http_ok "$TRIGGER_API_URL/healthcheck" 300 || fail "trigger.dev webapp did not become healthy, see podman logs $WEBAPP_CONTAINER"
log "trigger.dev webapp up"

ENCRYPTION_KEY="$(grep '^ENCRYPTION_KEY=' "$STACK_CONF" | cut -d= -f2)" TRIGGER_POSTGRES_CONTAINER="$POSTGRES_CONTAINER" \
  bun "$SCRIPTS/bootstrap.ts" >"$TRIGGER_CONF.next" 2>"$LOGS/bootstrap.log" || fail "bootstrap failed, see $LOGS/bootstrap.log"
mv "$TRIGGER_CONF.next" "$TRIGGER_CONF"
chmod 600 "$TRIGGER_CONF"
agent_env
log "local user, org and project ready: $TRIGGER_PROJECT_REF"

start_trigger_dev
start_agent_api

start_bg ui "$ROOT/ui" bun run dev --port "$(service_port ui)" --strictPort
wait_port_up "$(service_port ui)" 30 || fail "ui did not start, see $LOGS/ui.log"

log ""
log "ui                 $(service_url ui)"
log "agent-api          $(service_url agent-api)"
log "trigger.dev        $TRIGGER_API_URL (dashboard login: ./scripts/dashboard-login.sh)"
log "trigger.dev runs   $TRIGGER_API_URL/orgs/$TRIGGER_ORG_SLUG/projects/$TRIGGER_PROJECT_SLUG/env/dev/runs"
