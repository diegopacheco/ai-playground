#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

port_up "$(service_port mcp)" || fail "mcp is down, run scripts/start-all.sh"
MCP_PORT="$(service_port mcp)" "$ROOT/.venv/bin/calculator-client" "$@"
