#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "build all started"
"$SCRIPTS/build-frontend.sh"
log "build all done"
