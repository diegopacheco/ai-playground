#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "go unit tests"
(cd "$ROOT/backend" && go test ./...)
log "frontend unit tests"
(cd "$ROOT/frontend" && bun test src)
