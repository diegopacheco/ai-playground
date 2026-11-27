#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "backend unit tests"
(cd "$ROOT/backend" && uv run pytest tests --ignore=tests/integration -q)
log "frontend unit tests"
(cd "$ROOT/frontend" && bun test src)
