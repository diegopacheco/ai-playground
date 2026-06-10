#!/usr/bin/env bash
set -euo pipefail

CLAUDE_DEST="$HOME/.claude/skills/bundle-size"
CODEX_DEST="$HOME/.codex/skills/bundle-size"

if [ -d "$CLAUDE_DEST" ]; then
  rm -rf "$CLAUDE_DEST"
  echo "Removed bundle-size from Claude Code"
else
  echo "Claude Code skill not found - nothing to remove"
fi

if [ -d "$CODEX_DEST" ]; then
  rm -rf "$CODEX_DEST"
  echo "Removed bundle-size from Codex"
else
  echo "Codex skill not found - nothing to remove"
fi

echo "Done. bundle-size uninstalled."
