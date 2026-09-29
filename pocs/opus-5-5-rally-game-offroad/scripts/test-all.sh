#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "tests started"

require npm
( cd "$ROOT" && npm test ) || fail "game tests failed"

log "tests passed"
