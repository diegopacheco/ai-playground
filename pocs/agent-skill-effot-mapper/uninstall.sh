#!/usr/bin/env bash
set -euo pipefail

CLAUDE_DIR="$HOME/.claude/skills/effort-map"
CODEX_DIR="$HOME/.codex/skills/effort-map"
CODEX_PROMPT="$HOME/.codex/prompts/effort-map.md"

uninstall_claude() {
  rm -rf "$CLAUDE_DIR"
  echo "Removed Claude skill at $CLAUDE_DIR"
}

uninstall_codex() {
  rm -rf "$CODEX_DIR"
  rm -f "$CODEX_PROMPT"
  echo "Removed Codex skill at $CODEX_DIR"
}

echo "Uninstall the effort-map skill from:"
echo "  1) Claude"
echo "  2) Codex"
echo "  3) Both"
printf "Choose [1/2/3]: "
read -r choice

case "$choice" in
  1) uninstall_claude ;;
  2) uninstall_codex ;;
  3) uninstall_claude; uninstall_codex ;;
  *) echo "Invalid choice: $choice"; exit 1 ;;
esac

echo "Done."
