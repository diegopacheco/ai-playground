#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "setup started"

require node
require npm
npm install --no-audit --no-fund

for entry in "node_modules/" ".run/"; do
  if ! grep -qx "$entry" "$ROOT/.gitignore" 2>/dev/null; then
    printf "%s\n" "$entry" >>"$ROOT/.gitignore"
  fi
done

log "setup done"
