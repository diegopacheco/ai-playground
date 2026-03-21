#!/bin/bash
SKILL_DIR="$HOME/.claude/skills/infra-automation-generator"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_SKILL="$SCRIPT_DIR/skills/infra-automation-generator/SKILL.md"
if [ ! -f "$SOURCE_SKILL" ]; then
  echo "SKILL.md not found at $SOURCE_SKILL"
  exit 1
fi
mkdir -p "$SKILL_DIR"
cp "$SOURCE_SKILL" "$SKILL_DIR/SKILL.md"
echo "Installed infra-automation-generator skill to $SKILL_DIR"
echo ""
echo "Usage: type 'generate infra' or '/infra-automation-generator' in Claude Code or Codex"
