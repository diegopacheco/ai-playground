#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require node
require npm
log "installing npm dependencies"
npm install --no-fund --no-audit
if [ ! -d "$ROOT/node_modules/electron/dist/Electron.app" ]; then
  node "$ROOT/node_modules/electron/install.js"
fi
if [ -f "$ROOT/data/movies.json" ]; then
  log "data/movies.json already built"
else
  log "building data/movies.json from DataSF, Wikipedia and Wikidata"
  node "$ROOT/tools/build-data.mjs"
fi
log "setup done"
