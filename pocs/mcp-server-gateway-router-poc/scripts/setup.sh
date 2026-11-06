#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require node
require lsof
major="$(node -p 'process.versions.node.split(".")[0]')"
[ "$major" -ge 24 ] || fail "node 24 or newer is required, found $(node -v)"
log "node $(node -v) ok, no dependencies to install"
