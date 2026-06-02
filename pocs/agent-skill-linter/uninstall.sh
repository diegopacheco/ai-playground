#!/usr/bin/env bash
set -euo pipefail
rm -rf "$HOME/.claude/skills/agent-skill-linter"
rm -f "$HOME/.claude/commands/lint.md"
rm -f "$HOME/.claude/commands/lint-site.md"
echo "uninstalled agent-skill-linter"
