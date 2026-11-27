#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "tests started"
"$SCRIPTS/unit-test.sh"
"$SCRIPTS/integration-test.sh"
log "tests passed"
