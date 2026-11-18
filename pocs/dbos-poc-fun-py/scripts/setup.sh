#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require podman
require podman-compose
require uv
require bun
require curl
agent_env
require "$(printf "%s" "$AGENT_PROVIDER")"
(cd "$ROOT/agent-service" && uv sync)
(cd "$ROOT/ui" && bun install)
(cd "$ROOT/dashboard" && bun install)
podman-compose -f "$ROOT/podman-compose.yml" pull
log "setup done"
