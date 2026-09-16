#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require bun

if [ ! -f "$ROOT/packages/engine/dist/index.d.ts" ]; then
  "$SCRIPTS/setup.sh"
fi

log "running the engine and pilot suites"
bun test packages || fail "unit tests failed"

log "typechecking every workspace with typescript"
bun run --filter '*' typecheck || fail "typecheck failed"

log "building the arena"
bun run --filter '@fly-zord/web' build >"$LOGS/build.log" 2>&1 || fail "the arena build failed, see $LOGS/build.log"

log "all suites passed"
