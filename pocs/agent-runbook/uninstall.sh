#!/bin/bash

SKILL_NAME="runbook-generator"
SKILL_DIR="$HOME/.claude/skills/$SKILL_NAME"

if [ ! -d "$SKILL_DIR" ]; then
  echo "Skill '$SKILL_NAME' is not installed"
  exit 0
fi

rm -rf "$SKILL_DIR"
echo "Uninstalled '$SKILL_NAME' from $SKILL_DIR"
