#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "agent-service type check"
(cd "$ROOT/agent-service" && uv run ty check src tests) || fail "agent-service type check failed"
log "agent-service tests"
(cd "$ROOT/agent-service" && uv run pytest -q) || fail "agent-service tests failed"
log "ui typecheck and build"
(cd "$ROOT/ui" && bun run build) || fail "ui build failed"
log "ui tests"
(cd "$ROOT/ui" && bun test) || fail "ui tests failed"
log "dashboard typecheck and build"
(cd "$ROOT/dashboard" && bun run build) || fail "dashboard build failed"
log "dashboard tests"
(cd "$ROOT/dashboard" && bun test) || fail "dashboard tests failed"
log "end to end"
"$SCRIPTS/test-e2e.sh" || fail "end to end test failed"
log "all tests passed"
