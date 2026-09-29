#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require node
node --test "tests/*.test.mjs" || fail "node tests failed"
log "all tests passed"
