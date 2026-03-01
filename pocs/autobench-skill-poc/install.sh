#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLAUDE_DIR="$HOME/.claude"
SKILLS_DIR="$CLAUDE_DIR/skills/autobench"
COMMANDS_DIR="$CLAUDE_DIR/commands"

mkdir -p "$SKILLS_DIR"
mkdir -p "$COMMANDS_DIR"

cp "$SCRIPT_DIR/skills/autobench/SKILL.md" "$SKILLS_DIR/SKILL.md"
cp "$SCRIPT_DIR/commands/autobench.md" "$COMMANDS_DIR/autobench.md"

mkdir -p "$SKILLS_DIR/templates"
cp "$SCRIPT_DIR/templates/bench-csv.sh" "$SKILLS_DIR/templates/"
cp "$SCRIPT_DIR/templates/bench-crud.sh" "$SKILLS_DIR/templates/"
cp "$SCRIPT_DIR/templates/bench-uuid.sh" "$SKILLS_DIR/templates/"

chmod +x "$SKILLS_DIR/templates/"*.sh

echo "AutoBench skill installed."
echo "  Skill: $SKILLS_DIR/SKILL.md"
echo "  Command: $COMMANDS_DIR/autobench.md"
echo "  Templates: $SKILLS_DIR/templates/"
echo ""
echo "Usage: /autobench"
