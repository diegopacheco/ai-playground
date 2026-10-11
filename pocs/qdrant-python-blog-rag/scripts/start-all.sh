#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "starting"

require podman-compose
require bun
require_ollama
[ -x "$ROOT/backend/.venv/bin/python" ] || fail "backend is not installed, run ./scripts/setup.sh first"

podman-compose up -d
wait_port_up "$QDRANT_PORT" 60 || fail "qdrant did not open port $QDRANT_PORT"
wait_http "$QDRANT_URL/readyz" 60 || fail "qdrant is not ready on $QDRANT_URL"
log "qdrant up on $(service_url qdrant)"

start_bg backend "$ROOT/backend" .venv/bin/python -m blograg.server
wait_port_up "$API_PORT" 60 || fail "backend did not open port $API_PORT, see $LOGS/backend.log"
log "backend up on $(service_url backend)"

start_bg frontend "$ROOT/ui" bun run dev
wait_port_up "$UI_PORT" 60 || fail "frontend did not open port $UI_PORT, see $LOGS/frontend.log"
log "frontend up on $(service_url frontend)"

"$SCRIPTS/status.sh"

log "links"
for name in $(service_names); do
  printf "%-14s %s\n" "$name" "$(service_url "$name")"
done

stats="$(curl -fsS "$(service_url backend)")"
case "$stats" in
  *'"posts": 0,'*) log "the collection is empty, run ./scripts/ingest.sh to index the blog" ;;
esac
