#!/bin/bash

SKILL_NAME="leak-detector"
CLAUDE_DEST="$HOME/.claude/skills/$SKILL_NAME"
CODEX_DEST="$HOME/.codex/skills/$SKILL_NAME"

if [ -d "$CLAUDE_DEST" ]; then
    rm -rf "$CLAUDE_DEST"
    echo "Removed $SKILL_NAME from Claude Code"
else
    echo "Claude Code skill not found - nothing to remove"
fi

if [ -d "$CODEX_DEST" ]; then
    rm -rf "$CODEX_DEST"
    echo "Removed $SKILL_NAME from Codex"
else
    echo "Codex skill not found - nothing to remove"
fi

echo "Done. $SKILL_NAME uninstalled."
