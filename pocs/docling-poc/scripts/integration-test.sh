#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "docling engine integration tests"
(cd "$ROOT/backend" && uv run pytest tests/integration -q -p no:warnings)
