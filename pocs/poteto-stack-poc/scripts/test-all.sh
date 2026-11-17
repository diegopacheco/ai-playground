#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require bun
bun run typecheck || fail "typecheck failed"
bun test || fail "unit tests failed"
log "all tests passed"
