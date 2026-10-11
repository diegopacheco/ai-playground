#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require curl
port_up "$API_PORT" || fail "backend is not running on $API_PORT, run ./scripts/start-all.sh first"

body="{}"
if [ -n "${PRODUCT_LIMIT:-}" ]; then
  body="{\"limit\": $PRODUCT_LIMIT}"
fi

log "ingesting products, this can take a few minutes"
response="$(curl -fsS -X POST -H "Content-Type: application/json" -d "$body" "$API_URL/api/ingest")" || fail "ingest failed, see $LOGS/backend.log"

log "indexed $(printf "%s" "$response" | sed 's/.*"indexed":\([0-9]*\).*/\1/')"
log "skipped $(printf "%s" "$response" | sed 's/.*"skipped":\([0-9]*\).*/\1/')"
log "products in qdrant $(curl -fsS "$API_URL/api/stats" | sed 's/.*"products":\([0-9]*\).*/\1/')"
