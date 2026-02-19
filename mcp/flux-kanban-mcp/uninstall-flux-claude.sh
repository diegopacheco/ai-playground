#!/usr/bin/env bash
set -euo pipefail

USER_MCP="$HOME/.mcp.json"

if [[ -f "$USER_MCP" ]] && jq -e '.mcpServers.flux' "$USER_MCP" >/dev/null 2>&1; then
  tmp=$(mktemp)
  jq 'del(.mcpServers.flux)' "$USER_MCP" > "$tmp" && mv "$tmp" "$USER_MCP"
  echo "MCP server 'flux' removed from $USER_MCP"
else
  claude mcp remove flux 2>/dev/null || echo "MCP server 'flux' not found in any scope"
fi
