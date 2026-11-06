#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "backend tests"
(cd "$ROOT/backend" && bun test && ./node_modules/.bin/tsc --noEmit) || fail "backend tests failed"
log "frontend tests"
(cd "$ROOT/frontend" && bun test && ./node_modules/.bin/tsc --noEmit) || fail "frontend tests failed"
log "all tests passed"
