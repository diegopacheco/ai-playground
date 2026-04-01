#!/bin/bash
SKILL_DIR="$HOME/.claude/skills/metrics-report"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$SKILL_DIR"
cp "$SCRIPT_DIR/SKILL.md" "$SKILL_DIR/SKILL.md"

echo "metrics-report skill installed to $SKILL_DIR"
echo "You can now use /metrics-report in Claude Code"
