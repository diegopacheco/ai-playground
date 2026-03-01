#!/usr/bin/env bash
set -euo pipefail

CLAUDE_DIR="$HOME/.claude"
SKILLS_DIR="$CLAUDE_DIR/skills/autobench"
COMMANDS_DIR="$CLAUDE_DIR/commands"

rm -rf "$SKILLS_DIR"
rm -f "$COMMANDS_DIR/autobench.md"

echo "AutoBench skill uninstalled."
echo "  Removed: $SKILLS_DIR"
echo "  Removed: $COMMANDS_DIR/autobench.md"
