#!/usr/bin/env bash
set -euo pipefail

CLAUDE_DEST="$HOME/.claude/skills/habit-tracker"
CODEX_DEST="$HOME/.codex/skills/habit-tracker"

if [ -d "$CLAUDE_DEST" ]; then
  rm -rf "$CLAUDE_DEST"
  echo "Removed habit-tracker from Claude Code"
else
  echo "Claude Code skill not found - nothing to remove"
fi

if [ -d "$CODEX_DEST" ]; then
  rm -rf "$CODEX_DEST"
  echo "Removed habit-tracker from Codex"
else
  echo "Codex skill not found - nothing to remove"
fi

echo "Done. habit-tracker uninstalled."
