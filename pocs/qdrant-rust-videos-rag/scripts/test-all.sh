#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require cargo
require bun

log "backend tests"
( cd "$ROOT/backend" && cargo test )

log "ui typecheck"
( cd "$ROOT/ui" && bun run typecheck )

log "all tests passed"
