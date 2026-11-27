#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require uv
require bun
log "installing backend dependencies"
(cd "$ROOT/backend" && uv sync)
log "downloading docling models"
(cd "$ROOT/backend" && uv run docling-tools models download layout tableformer >/dev/null)
log "installing frontend dependencies"
(cd "$ROOT/frontend" && bun install)
log "setup done"
