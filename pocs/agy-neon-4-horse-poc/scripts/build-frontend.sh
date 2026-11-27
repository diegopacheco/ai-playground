#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "building frontend"
require node

[ -f "$ROOT/node_modules/three/build/three.module.js" ] || fail "three is not installed, run ./scripts/setup.sh first"

node --check "$ROOT/server.mjs" || fail "syntax check failed for server.mjs"
for file in "$ROOT"/public/js/*.js; do
  node --check "$file" || fail "syntax check failed for $file"
done

log "frontend build passed"
