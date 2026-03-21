#!/bin/bash

SKILL_DIR="$HOME/.claude/skills/threat-analyst"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$SKILL_DIR"
cp "$SCRIPT_DIR/SKILL.md" "$SKILL_DIR/SKILL.md"

echo "threat-analyst skill installed to $SKILL_DIR"
echo "Usage: /threat-analyst"
