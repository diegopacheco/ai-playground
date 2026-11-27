#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "unit tests started"
require node
( cd "$ROOT" && node --test test/unit.test.mjs ) || fail "unit tests failed"
log "unit tests passed"
