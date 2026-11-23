#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "tests started"

require bun
bun test server || fail "server tests failed"
bun run typecheck || fail "typecheck failed"
API_PORT="$(service_port backend)" WEB_PORT="$(service_port frontend)" bun run build || fail "web build failed"

log "tests passed"
