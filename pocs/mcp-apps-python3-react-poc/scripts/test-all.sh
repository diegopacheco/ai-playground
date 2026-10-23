#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "backend: ty type check"
(cd "$ROOT/backend" && uv run ty check src tests) || fail "ty found type errors"

log "backend: pytest"
(cd "$ROOT/backend" && uv run pytest -q) || fail "backend tests failed"

log "frontend: tsc type check"
(cd "$ROOT/frontend" && bun run typecheck) || fail "tsc found type errors"

log "frontend: bun test"
(cd "$ROOT/frontend" && bun test) || fail "frontend tests failed"

log "all tests passed"
