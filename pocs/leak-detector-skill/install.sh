#!/bin/bash

SKILL_NAME="leak-detector"
SKILL_SRC="$(cd "$(dirname "$0")" && pwd)/skills/$SKILL_NAME"
CLAUDE_DEST="$HOME/.claude/skills/$SKILL_NAME"
CODEX_DEST="$HOME/.codex/skills/$SKILL_NAME"

if [ ! -f "$SKILL_SRC/SKILL.md" ]; then
    echo "ERROR: SKILL.md not found at $SKILL_SRC"
    exit 1
fi

mkdir -p "$CLAUDE_DEST"
cp "$SKILL_SRC/SKILL.md" "$CLAUDE_DEST/SKILL.md"
echo "Installed $SKILL_NAME to Claude Code: $CLAUDE_DEST"

if [ -d "$HOME/.codex" ]; then
    mkdir -p "$CODEX_DEST"
    cp "$SKILL_SRC/SKILL.md" "$CODEX_DEST/SKILL.md"
    echo "Installed $SKILL_NAME to Codex: $CODEX_DEST"
else
    echo "Codex not found at $HOME/.codex - skipping Codex install"
fi

echo "Done. Use /leak-detect to run the skill."
