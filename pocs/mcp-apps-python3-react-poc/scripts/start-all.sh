#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

[ -f "$ROOT/frontend/dist/view.html" ] || fail "frontend is not built, run ./scripts/setup.sh first"

port="$(service_port calculator)"
public_host="$(public_host)"
log "starting"
MCP_PUBLIC_HOST="$public_host" start_bg calculator "$ROOT/backend" uv run granian --interface asgi --access-log --host 127.0.0.1 --port "$port" calculator_app.app:app
wait_port_up "$port" 60 || fail "calculator did not open port $port, see $LOGS/calculator.log"

"$SCRIPTS/status.sh"

log "links"
printf "%-14s %s\n" host "$(service_url calculator /)"
printf "%-14s %s\n" mcp "$(service_url calculator /mcp)"
printf "%-14s %s\n" health "$(service_url calculator /health)"
if [ -n "$public_host" ]; then
  printf "%-14s %s\n" chatgpt "$(public_url "$public_host")"
fi
