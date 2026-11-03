#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require claude
port_up "$(service_port backend)" || fail "backend is down, run ./scripts/start-all.sh first"

log "asking claude code to deploy payments to prod, watch the Live (push) tab at $(service_url ui /)"
claude -p "Use the mcp-events MCP server. Deploy the service payments to prod with deploy_service. Then call poll_events for deploy.status_changed with the cursor it returned and waitMs 5000, keep the newest cursor and repeat until the status is deployed or failed." \
  --allowedTools "mcp__${MCP_NAME}__deploy_service,mcp__${MCP_NAME}__poll_events"
