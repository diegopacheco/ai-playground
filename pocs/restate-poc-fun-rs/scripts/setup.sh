#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require podman
require podman-compose
require cargo
require bun
require curl
agent_env
require "$AGENT_PROVIDER"
(cd "$ROOT/agent-service" && cargo build --release) || fail "agent-service release build failed"
(cd "$ROOT/ui" && bun install) || fail "ui install failed"
compose pull >"$LOGS/podman-compose.log" 2>&1 || fail "podman-compose pull failed, see $LOGS/podman-compose.log"
log "setup done"
