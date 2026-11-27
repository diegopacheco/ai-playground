#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "integration tests started"
require node
( cd "$ROOT" && node --test test/integration.test.mjs ) || fail "integration tests failed"
log "integration tests passed"
