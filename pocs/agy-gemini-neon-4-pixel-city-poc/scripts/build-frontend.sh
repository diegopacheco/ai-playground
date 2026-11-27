#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "building frontend"
require bun
( cd "$ROOT" && bun run build ) || fail "frontend build failed"
log "frontend build passed"
