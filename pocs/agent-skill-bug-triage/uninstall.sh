#!/usr/bin/env bash
set -euo pipefail

NAME="bug-triage"
CLAUDE_DEST="$HOME/.claude/skills/$NAME"
CODEX_DEST="$HOME/.codex/skills/$NAME"

TARGET="${1:-}"
if [ -z "$TARGET" ]; then
  echo "🧹 bug-triage uninstaller"
  echo ""
  echo "  1) 🤖 Claude Code   ($CLAUDE_DEST)"
  echo "  2) 🧠 Codex         ($CODEX_DEST)"
  echo "  3) 💥 Both"
  echo ""
  printf "👉 remove from where? [1/2/3] "
  read -r CHOICE
  case "$CHOICE" in
    1) TARGET="--claude" ;;
    2) TARGET="--codex" ;;
    3) TARGET="--both" ;;
    *) echo "❌ pick 1, 2 or 3"; exit 1 ;;
  esac
fi

remove_from() {
  local dest="$1"
  local label="$2"
  if [ -d "$dest" ]; then
    rm -rf "$dest"
    echo "🗑️  removed from $label: $dest"
  else
    echo "ℹ️  nothing installed for $label"
  fi
}

case "$TARGET" in
  --claude) remove_from "$CLAUDE_DEST" "Claude Code" ;;
  --codex) remove_from "$CODEX_DEST" "Codex" ;;
  --both)
    remove_from "$CLAUDE_DEST" "Claude Code"
    remove_from "$CODEX_DEST" "Codex"
    ;;
  *)
    echo "❌ usage: ./uninstall.sh [--claude|--codex|--both]"
    exit 1
    ;;
esac

echo "👋 done."
