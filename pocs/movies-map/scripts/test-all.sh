#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require node
log "checking desktop app syntax"
node --check desktop/main.cjs
node --check desktop/preload.cjs
log "running unit and api tests"
node --test "tests/*.test.mjs"
