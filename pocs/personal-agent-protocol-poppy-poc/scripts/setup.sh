#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require uv
require bun

log "installing backend dependencies with uv on python 3.15"
(cd "$ROOT/backend" && uv sync --python 3.15) || fail "uv sync failed"

log "installing frontend dependencies with bun"
(cd "$ROOT/frontend" && bun install) || fail "bun install failed"

log "setup done"
