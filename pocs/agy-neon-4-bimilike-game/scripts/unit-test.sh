#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "unit tests started"
require bun
bun test || fail "unit tests failed"
log "unit tests passed"
