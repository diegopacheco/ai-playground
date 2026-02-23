#!/bin/bash
echo "=== GraphQL-Postgres MCP Uninstaller ==="

echo ""
echo "--- Claude Code ---"
if command -v claude &> /dev/null; then
  claude mcp remove graph-postgres-mcp -s user 2>/dev/null
  echo "Removed graph-postgres-mcp from Claude Code."
else
  echo "SKIP: 'claude' CLI not found."
fi

echo ""
echo "--- Codex CLI ---"
CODEX_CONFIG="$HOME/.codex/config.toml"
if [ -f "$CODEX_CONFIG" ]; then
  sed -i.bak '/\[mcp-servers\.graph-postgres-mcp\]/,/^$/d' "$CODEX_CONFIG"
  sed -i.bak '/\[mcp-servers\.graph-postgres-mcp\.env\]/,/^$/d' "$CODEX_CONFIG"
  rm -f "${CODEX_CONFIG}.bak"
  echo "Removed graph-postgres-mcp from Codex CLI."
else
  echo "SKIP: ~/.codex/config.toml not found."
fi

echo ""
echo "=== Uninstall complete ==="
