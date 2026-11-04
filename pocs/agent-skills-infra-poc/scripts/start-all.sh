#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

[ -f "$INFRA/.secrets/clients.json" ] || fail "run scripts/setup.sh first"

compose_env
log "starting postgres, loki and grafana"
podman-compose up -d >"$LOGS/compose-up.log" 2>&1 || fail "podman-compose up failed, see $LOGS/compose-up.log"
wait_postgres || fail "postgres did not become ready"
podman exec -i agentinfra-postgres psql -q -U agentinfra -d agentinfra -v ON_ERROR_STOP=1 <"$INFRA/containers/postgres/migrate.sql" >"$LOGS/migrate.log" 2>&1 || fail "migration failed, see $LOGS/migrate.log"
wait_http_ok "$(service_url loki)" || fail "loki did not become ready"

for name in $API_SERVICES; do
  start_bg "$name" "$INFRA" env PYTHONPATH="$INFRA" "$INFRA/.venv/bin/granian" --interface asgi --host 127.0.0.1 --port "$(service_port "$name")" --log-level warning "$name.app:app"
done
start_bg ui "$UI_DIR" bun x vite --port "$(service_port ui)" --strictPort --host 127.0.0.1

for name in $API_SERVICES; do
  wait_http_ok "$(service_url "$name")/health" || fail "$name did not start, see $LOGS/$name.log"
done
for name in ui grafana; do
  wait_port_up "$(service_port "$name")" || fail "$name did not start"
done

log ""
for name in $(service_names); do
  log "$(printf "%-9s %s" "$name" "$(service_url "$name")")"
done
