#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "setup started"

require node
major="$(node -p 'process.versions.node.split(".")[0]')"
[ "$major" -ge 20 ] || fail "node 20 or newer is required, found $(node --version)"

if ! grep -q '^\.run/$' "$ROOT/.gitignore" 2>/dev/null; then
  printf ".run/\n" >>"$ROOT/.gitignore"
fi

log "setup done"
