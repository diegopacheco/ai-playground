#!/bin/bash
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENTRY_POINT="$PROJECT_DIR/dist/index.js"

echo "=== GraphQL-Postgres MCP Installer ==="
echo "Project: $PROJECT_DIR"

cd "$PROJECT_DIR"
npm install
npm run build

if [ ! -f "$ENTRY_POINT" ]; then
  echo "ERROR: Build failed. $ENTRY_POINT not found."
  exit 1
fi

echo ""
echo "--- Claude Code ---"
if command -v claude &> /dev/null; then
  claude mcp remove graph-postgres-mcp -s user 2>/dev/null
  claude mcp add graph-postgres-mcp -s user -- node "$ENTRY_POINT"
  echo "Registered graph-postgres-mcp in Claude Code (user scope)."
else
  echo "SKIP: 'claude' CLI not found."
fi

echo ""
echo "--- Codex CLI ---"
CODEX_CONFIG="$HOME/.codex/config.toml"
if [ -d "$HOME/.codex" ]; then
  if [ -f "$CODEX_CONFIG" ]; then
    sed -i.bak '/\[mcp-servers\.graph-postgres-mcp\]/,/^$/d' "$CODEX_CONFIG"
    rm -f "${CODEX_CONFIG}.bak"
  fi
  cat >> "$CODEX_CONFIG" <<EOF

[mcp-servers.graph-postgres-mcp]
type = "stdio"
command = "node"
args = ["$ENTRY_POINT"]

[mcp-servers.graph-postgres-mcp.env]
PG_HOST = "localhost"
PG_PORT = "5432"
PG_USER = "graphmcp"
PG_PASSWORD = "graphmcp123"
PG_DATABASE = "graphmcpdb"
EOF
  echo "Registered graph-postgres-mcp in Codex CLI."
else
  echo "SKIP: ~/.codex directory not found."
fi

echo ""
echo "=== Installation complete ==="
