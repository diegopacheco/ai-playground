#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "backend: ty"
(cd "$ROOT/backend" && uv run ty check) || fail "backend type check failed"
log "backend: pytest"
(cd "$ROOT/backend" && uv run pytest -q) || fail "backend tests failed"
log "frontend: tsc"
(cd "$ROOT/frontend" && bun run typecheck) || fail "frontend type check failed"
log "frontend: bun test"
(cd "$ROOT/frontend" && bun test) || fail "frontend tests failed"
log "all tests passed"
