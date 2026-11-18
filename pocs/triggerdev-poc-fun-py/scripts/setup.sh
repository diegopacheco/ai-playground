#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require podman
require podman-compose
require uv
require bun
require node
require curl
require openssl
agent_env
require "$(printf "%s" "$AGENT_PROVIDER")"
(cd "$ROOT/api" && uv sync)
(cd "$ROOT/tasks" && bun install)
(cd "$ROOT/ui" && bun install)
ensure_stack_conf
compose pull
log "setup done"
