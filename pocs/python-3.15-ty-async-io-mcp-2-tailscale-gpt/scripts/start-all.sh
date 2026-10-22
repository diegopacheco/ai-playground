#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

port="$(service_port mcp)"
public_host=""
if [ -f "$SCRIPTS/tailscale.env" ]; then
  public_host="$(grep '^MCP_PUBLIC_HOST=' "$SCRIPTS/tailscale.env" | cut -d= -f2 || true)"
fi
if port_up "$port"; then
  log "mcp already listening on $port"
else
  MCP_PORT="$port" MCP_PUBLIC_HOST="$public_host" start_bg mcp "$ROOT/.venv/bin/calculator-server"
  wait_port_up "$port" 60 || fail "mcp did not start, see $LOGS/mcp.log"
fi
log "mcp: $(service_url mcp)"
