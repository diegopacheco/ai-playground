#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require podman-compose
require bun
require curl
require_ollama
use_java25

[ -f "$JAR" ] || fail "backend jar not found, run ./scripts/setup.sh first"
[ -d "$ROOT/ui/node_modules" ] || fail "ui dependencies not installed, run ./scripts/setup.sh first"

log "starting qdrant"
podman-compose up -d
wait_port_up "$QDRANT_PORT" 60 || fail "qdrant did not open port $QDRANT_PORT"
wait_http "$QDRANT_URL/readyz" 60 || fail "qdrant is not ready on $QDRANT_URL"

start_bg backend "$ROOT/backend" java -jar "$JAR"
wait_port_up "$API_PORT" 60 || fail "backend did not open port $API_PORT, see $LOGS/backend.log"
wait_http "$API_URL/api/stats" 60 || fail "backend is not answering on $API_URL, see $LOGS/backend.log"

start_bg frontend "$ROOT/ui" bun run dev
wait_port_up "$UI_PORT" 60 || fail "frontend did not open port $UI_PORT, see $LOGS/frontend.log"

"$SCRIPTS/status.sh"

for name in $(service_names); do
  printf "%-10s %s\n" "$name" "$(service_url "$name")"
done

if curl -fsS "$API_URL/api/stats" | grep -q '"products":0[,}]'; then
  log "collection is empty, run ./scripts/ingest.sh to index the products"
fi
