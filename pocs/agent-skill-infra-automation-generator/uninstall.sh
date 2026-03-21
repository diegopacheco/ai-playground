#!/bin/bash
SKILL_DIR="$HOME/.claude/skills/infra-automation-generator"
if [ ! -d "$SKILL_DIR" ]; then
  echo "Skill not found at $SKILL_DIR - nothing to uninstall"
  exit 0
fi
rm -rf "$SKILL_DIR"
echo "Removed infra-automation-generator skill from $SKILL_DIR"
