#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "building frontend"
require bun
bun run build || fail "frontend build failed"
log "frontend built"
