#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require node

start_service() {
  local name
  name="$1"
  shift
  start_bg "$name" env PORT="$(service_port "$name")" "$@"
  wait_port_up "$(service_port "$name")" 30 || fail "$name did not open port $(service_port "$name"), see $LOGS/$name.log"
  log "$name up on $(service_url "$name")"
}

start_service math node mcp-servers/math-mcp/server.js
start_service text node mcp-servers/text-mcp/server.js
start_service gateway env MCP_DIR="$ROOT/dynamic-mcps" UPSTREAMS="math=$(service_url math),text=$(service_url text)" node gateway.js

"$SCRIPTS/status.sh"

log "links"
for name in $(service_names); do
  printf "%-10s %s\n" "$name" "$(service_url "$name")"
done
log "connect a client to the gateway only: claude mcp add --transport http gateway $(service_url gateway)"
