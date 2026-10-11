#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require podman-compose
require bun
require_ollama
[ -x "$BIN" ] || fail "backend is not built, run ./scripts/setup.sh first"
[ -d "$ROOT/ui/node_modules" ] || fail "ui dependencies are missing, run ./scripts/setup.sh first"

log "starting qdrant"
podman-compose up -d
wait_port_up "$QDRANT_PORT" 60 || fail "qdrant did not open port $QDRANT_PORT"
wait_http "$QDRANT_URL/readyz" 60 || fail "qdrant is not ready on $QDRANT_URL"
wait_port_up "$QDRANT_GRPC_PORT" 60 || fail "qdrant did not open grpc port $QDRANT_GRPC_PORT"

start_bg backend "$ROOT/backend" "$BIN" serve
wait_port_up "$API_PORT" 60 || fail "backend did not open port $API_PORT, see $LOGS/backend.log"

start_bg frontend "$ROOT/ui" bun run dev
wait_port_up "$UI_PORT" 60 || fail "frontend did not open port $UI_PORT, see $LOGS/frontend.log"

"$SCRIPTS/status.sh"

for name in $(service_names); do
  printf "%-14s %s\n" "$name" "$(service_url "$name")"
done

stats="$(curl -fsS "$(service_url backend)")" || fail "backend stats failed, see $LOGS/backend.log"
case "$stats" in
  *'"videos":0,'*|*'"videos":0}'*) log "the collection is empty, run ./scripts/ingest.sh to index the channel" ;;
esac
