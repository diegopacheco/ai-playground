#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "agent-api type check"
(cd "$ROOT/api" && uv run ty check src tests) || fail "agent-api type check failed"
log "agent-api tests"
(cd "$ROOT/api" && uv run pytest -q) || fail "agent-api tests failed"
log "tasks typecheck"
(cd "$ROOT/tasks" && bunx tsc --noEmit) || fail "tasks typecheck failed"
log "tasks tests"
(cd "$ROOT/tasks" && bun test) || fail "tasks tests failed"
log "ui typecheck and build"
(cd "$ROOT/ui" && bun run build) || fail "ui build failed"
log "ui tests"
(cd "$ROOT/ui" && bun test) || fail "ui tests failed"
log "end to end"
"$SCRIPTS/test-e2e.sh" || fail "end to end test failed"
log "all tests passed"
