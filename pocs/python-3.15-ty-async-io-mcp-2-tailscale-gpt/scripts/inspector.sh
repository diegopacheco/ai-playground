#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require npx
port_up "$(service_port mcp)" || "$SCRIPTS/start-all.sh"
port_up "$(service_port inspector)" && fail "inspector port $(service_port inspector) is already in use"
CLIENT_PORT="$(service_port inspector)" exec npx -y @modelcontextprotocol/inspector@latest \
  --web --transport http --server-url "$(service_url mcp)"
