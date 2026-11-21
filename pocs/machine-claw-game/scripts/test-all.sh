#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "tests started"

require bun
bun run typecheck || fail "typecheck failed"
bun test || fail "tests failed"
bun run build || fail "build failed"

log "tests passed"
