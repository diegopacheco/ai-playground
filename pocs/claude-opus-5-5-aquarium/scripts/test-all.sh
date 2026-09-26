#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require node
log "unit tests"
node --test "$ROOT"/tests/*.test.mjs || fail "unit tests failed"
log "tests passed"
