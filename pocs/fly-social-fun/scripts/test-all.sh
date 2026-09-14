#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "tests started"

require npm
[ -d "$ROOT/node_modules/three" ] || fail "dependencies missing, run ./scripts/setup.sh first"
npm test || fail "tests failed"

log "tests passed"
