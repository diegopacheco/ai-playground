#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "setup started"

require go
require bun

( cd "$ROOT/backend" && go mod download && go build -o "$RUN/bin/api" ./cmd/server ) || fail "backend build failed"
( cd "$ROOT/frontend" && bun install ) || fail "frontend install failed"
( cd "$ROOT/desktop" && bun install ) || fail "desktop install failed"
[ -d "$ROOT/desktop/node_modules/electron/dist/Electron.app" ] || ( cd "$ROOT/desktop" && bun node_modules/electron/install.js )
[ -d "$ROOT/desktop/node_modules/electron/dist/Electron.app" ] || fail "electron binary download failed"

mkdir -p "$ROOT/data"

log "setup done"
