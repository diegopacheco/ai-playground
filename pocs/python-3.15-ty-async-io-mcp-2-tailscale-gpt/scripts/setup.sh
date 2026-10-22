#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require uv
require lsof
uv sync || fail "uv sync failed"
log "setup done"
