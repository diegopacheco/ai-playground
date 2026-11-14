#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "tests started"
require bun
bunx tsc --noEmit || fail "typecheck failed"
bun test src || fail "unit tests failed"
bunx vite build || fail "build failed"
log "tests passed"
