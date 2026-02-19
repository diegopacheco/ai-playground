#!/usr/bin/env bash
set -euo pipefail

if ! command -v claude >/dev/null 2>&1; then
  echo "claude cli is required. Install Claude Code: https://claude.ai/code" >&2
  exit 1
fi

claude mcp remove flux

echo "MCP server 'flux' removed from Claude Code"
