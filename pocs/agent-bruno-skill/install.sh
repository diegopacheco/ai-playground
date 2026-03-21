#!/bin/bash

SKILL_NAME="bruno-generator"
SKILL_DIR="$HOME/.claude/skills/$SKILL_NAME"
SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -d "$SKILL_DIR" ]; then
  echo "Skill '$SKILL_NAME' already installed at $SKILL_DIR"
  echo "Updating..."
  rm -rf "$SKILL_DIR"
fi

mkdir -p "$SKILL_DIR"
cp "$SOURCE_DIR/SKILL.md" "$SKILL_DIR/SKILL.md"

echo "Installed '$SKILL_NAME' to $SKILL_DIR"
echo "You can now use /bruno-generator in Claude Code"
