#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_ollama
port_up "$QDRANT_PORT" || fail "qdrant is not running on $QDRANT_PORT, run ./scripts/start-all.sh first"

log "indexing every post into qdrant"
( cd "$ROOT/backend" && .venv/bin/python -m blograg.ingest ) || fail "ingest failed"
