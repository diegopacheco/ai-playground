#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "type checking backend"
(cd "$ROOT/backend" && uv run ty check app tests)
log "backend built"
