#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "building frontend"
(cd "$ROOT/frontend" && bun run build)
log "frontend built"
