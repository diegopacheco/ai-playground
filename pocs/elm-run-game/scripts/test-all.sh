#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "tests started"

require npm
[ -d "$ROOT/node_modules" ] || fail "dependencies missing, run ./scripts/setup.sh first"
npm test || fail "elm tests failed"

log "tests passed"
