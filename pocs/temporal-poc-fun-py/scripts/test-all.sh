#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "service type check"
(cd "$ROOT/service" && uv run ty check src tests) || fail "service type check failed"
log "service tests"
(cd "$ROOT/service" && uv run pytest -q) || fail "service tests failed"
log "ui typecheck and build"
(cd "$ROOT/ui" && bun run build) || fail "ui build failed"
log "ui tests"
(cd "$ROOT/ui" && bun test) || fail "ui tests failed"
log "end to end"
"$SCRIPTS/test-e2e.sh" || fail "end to end test failed"
log "all tests passed"
