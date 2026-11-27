#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

"$SCRIPTS/build-backend.sh"
"$SCRIPTS/build-frontend.sh"
log "all built"
