#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require node
require npm
npm install --no-audit --no-fund
[ -d "$ROOT/node_modules/electron/dist/Electron.app" ] || node "$ROOT/node_modules/electron/install.js"
[ -d "$ROOT/node_modules/electron/dist/Electron.app" ] || fail "electron binary download failed"
node tools/fetch-logos.mjs
log "setup done"
