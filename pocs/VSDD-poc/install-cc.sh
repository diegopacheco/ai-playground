#!/bin/bash
SKILL_DIR="$(cd "$(dirname "$0")/agent-skill" && pwd)"
TARGET="$HOME/.claude/skills/vsdd"
mkdir -p "$HOME/.claude/skills"
if [ -L "$TARGET" ] || [ -e "$TARGET" ]; then
  rm -rf "$TARGET"
fi
ln -s "$SKILL_DIR" "$TARGET"
echo "VSDD skill installed: $TARGET -> $SKILL_DIR"
