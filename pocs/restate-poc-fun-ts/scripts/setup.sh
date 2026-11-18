#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require podman
require podman-compose
require bun
require curl
agent_env
require "$(printf "%s" "$AGENT_PROVIDER")"
(cd "$ROOT/agent-service" && bun install)
(cd "$ROOT/ui" && bun install)
podman-compose -f "$ROOT/podman-compose.yml" pull
log "setup done"
