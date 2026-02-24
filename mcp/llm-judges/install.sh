#!/bin/bash
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENTRY_POINT="$PROJECT_DIR/target/release/llm-judges"

echo "=== LLM Judges MCP Installer ==="
echo "Project: $PROJECT_DIR"

cd "$PROJECT_DIR"
cargo build --release

if [ ! -f "$ENTRY_POINT" ]; then
  echo "ERROR: Build failed. $ENTRY_POINT not found."
  exit 1
fi

echo ""
echo "--- Claude Code ---"
if command -v claude &> /dev/null; then
  claude mcp remove llm-judges -s user 2>/dev/null
  claude mcp add llm-judges -s user -- "$ENTRY_POINT"
  echo "Registered llm-judges in Claude Code (user scope)."
else
  echo "SKIP: 'claude' CLI not found."
fi

echo ""
echo "--- Codex CLI ---"
CODEX_CONFIG="$HOME/.codex/config.toml"
if [ -d "$HOME/.codex" ]; then
  if [ -f "$CODEX_CONFIG" ]; then
    sed -i.bak '/\[mcp-servers\.llm-judges\]/,/^$/d' "$CODEX_CONFIG"
    sed -i.bak '/\[mcp_servers\.llm-judges\]/,/^$/d' "$CODEX_CONFIG"
    rm -f "${CODEX_CONFIG}.bak"
  fi
  cat >> "$CODEX_CONFIG" <<EOF

[mcp_servers.llm-judges]
type = "stdio"
command = "$ENTRY_POINT"
args = []
EOF
  echo "Registered llm-judges in Codex CLI."
else
  echo "SKIP: ~/.codex directory not found."
fi

echo ""
echo "=== Installation complete ==="
