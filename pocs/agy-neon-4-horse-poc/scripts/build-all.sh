#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "build started"
"$SCRIPTS/build-frontend.sh"
log "build passed"
