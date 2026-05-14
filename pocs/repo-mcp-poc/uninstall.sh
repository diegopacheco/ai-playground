#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/.mcp/repo-tool"
CLAUDE_CONFIG="$HOME/.claude.json"

if [ -d "$ROOT" ]; then
  rm -rf "$ROOT"
  echo "Removed $ROOT"
fi

if [ -f "$CLAUDE_CONFIG" ]; then
  node -e "
    const fs = require('fs');
    const p = '$CLAUDE_CONFIG';
    let cfg;
    try { cfg = JSON.parse(fs.readFileSync(p, 'utf8')); } catch (e) { process.exit(0); }
    if (cfg.mcpServers && cfg.mcpServers['repo-mcp']) {
      delete cfg.mcpServers['repo-mcp'];
      fs.writeFileSync(p, JSON.stringify(cfg, null, 2));
      console.log('Unregistered repo-mcp from ' + p);
    }
  "
fi
