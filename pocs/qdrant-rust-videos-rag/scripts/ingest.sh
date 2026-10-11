#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_ollama
[ -x "$BIN" ] || fail "backend is not built, run ./scripts/setup.sh first"
port_up "$QDRANT_GRPC_PORT" || fail "qdrant is not running on $QDRANT_GRPC_URL, run ./scripts/start-all.sh first"

log "ingesting channel videos into qdrant"
"$BIN" ingest
