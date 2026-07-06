#!/usr/bin/env bash
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)/effort-map"
CLAUDE_DIR="$HOME/.claude/skills/effort-map"
CODEX_DIR="$HOME/.codex/skills/effort-map"
CODEX_PROMPT="$HOME/.codex/prompts/effort-map.md"

install_claude() {
  mkdir -p "$HOME/.claude/skills"
  rm -rf "$CLAUDE_DIR"
  cp -R "$SRC" "$CLAUDE_DIR"
  echo "Installed for Claude at $CLAUDE_DIR"
}

install_codex() {
  mkdir -p "$HOME/.codex/skills" "$HOME/.codex/prompts"
  rm -rf "$CODEX_DIR"
  cp -R "$SRC" "$CODEX_DIR"
  printf '%s\n' "Run the effort-map skill: read $CODEX_DIR/SKILL.md and follow every step exactly." > "$CODEX_PROMPT"
  echo "Installed for Codex at $CODEX_DIR"
  echo "Registered Codex command at $CODEX_PROMPT"
}

echo "Install the effort-map skill for:"
echo "  1) Claude"
echo "  2) Codex"
echo "  3) Both"
printf "Choose [1/2/3]: "
read -r choice

case "$choice" in
  1) install_claude ;;
  2) install_codex ;;
  3) install_claude; install_codex ;;
  *) echo "Invalid choice: $choice"; exit 1 ;;
esac

echo "Done. Run /effort-map inside any project."
