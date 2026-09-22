#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "starting"
require "$PYTHON"
require npx

start_bg mcp "$ROOT/mcp" env MCP_PORT="$(service_port mcp)" "$PYTHON" math_server.py
wait_port_up "$(service_port mcp)" 30 || fail "mcp did not open port $(service_port mcp), see $LOGS/mcp.log"
log "mcp up on $(service_url mcp)/mcp"

start_bg bridge "$ROOT/bridge" env BRIDGE_PORT="$(service_port bridge)" "$PYTHON" bridge.py
wait_port_up "$(service_port bridge)" 30 || fail "bridge did not open port $(service_port bridge), see $LOGS/bridge.log"
log "bridge up on $(service_url bridge)"

mkdir -p "$RUN/bifrost"
sed -e "s/__BRIDGE_PORT__/$(service_port bridge)/g" -e "s/__MCP_PORT__/$(service_port mcp)/g" "$ROOT/bifrost/config.template.json" >"$RUN/bifrost/config.json"
start_bg bifrost "$ROOT" npx -y "$BIFROST_PACKAGE" --transport-version "$BIFROST_VERSION" -app-dir "$RUN/bifrost" -host 127.0.0.1 -port "$(service_port bifrost)"
wait_port_up "$(service_port bifrost)" 60 || fail "bifrost did not open port $(service_port bifrost), see $LOGS/bifrost.log"
log "bifrost up on $(service_url bifrost)"

start_bg app "$ROOT/app" env APP_PORT="$(service_port app)" BIFROST_URL="http://127.0.0.1:$(service_port bifrost)" MCP_URL="http://127.0.0.1:$(service_port mcp)" "$PYTHON" app.py
wait_port_up "$(service_port app)" 30 || fail "app did not open port $(service_port app), see $LOGS/app.log"
log "app up on $(service_url app)"

"$SCRIPTS/status.sh"

log "links"
printf "%-14s %s\n" "app" "$(service_url app)"
printf "%-14s %s\n" "bifrost-ui" "$(service_url bifrost)"
printf "%-14s %s\n" "bifrost-api" "$(service_url bifrost)/v1/chat/completions"
printf "%-14s %s\n" "bifrost-mcp" "$(service_url bifrost)/workspace/mcp-registry"
printf "%-14s %s\n" "bridge" "$(service_url bridge)/health"
printf "%-14s %s\n" "mcp" "$(service_url mcp)/mcp"
