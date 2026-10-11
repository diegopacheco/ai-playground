#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require mvn
require bun
use_java25

log "backend tests"
( cd "$ROOT/backend" && mvn -q test )

log "ui typecheck"
( cd "$ROOT/ui" && bun run typecheck )

log "all tests passed"
