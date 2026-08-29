#!/usr/bin/env bash
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)/skill"
NAME="bug-triage"
CLAUDE_DEST="$HOME/.claude/skills/$NAME"
CODEX_DEST="$HOME/.codex/skills/$NAME"

if ! command -v node >/dev/null 2>&1; then
  echo "❌ node is required"
  exit 1
fi
if [ ! -f "$SRC/SKILL.md" ]; then
  echo "❌ SKILL.md not found at $SRC"
  exit 1
fi

TARGET="${1:-}"
if [ -z "$TARGET" ]; then
  echo "🐛 bug-triage installer"
  echo ""
  echo "  1) 🤖 Claude Code   ($CLAUDE_DEST)"
  echo "  2) 🧠 Codex         ($CODEX_DEST)"
  echo "  3) 🚀 Both"
  echo ""
  printf "👉 install where? [1/2/3] "
  read -r CHOICE
  case "$CHOICE" in
    1) TARGET="--claude" ;;
    2) TARGET="--codex" ;;
    3) TARGET="--both" ;;
    *) echo "❌ pick 1, 2 or 3"; exit 1 ;;
  esac
fi

install_to() {
  local dest="$1"
  local label="$2"
  rm -rf "$dest"
  mkdir -p "$dest"
  cp -R "$SRC/." "$dest"
  chmod +x "$dest/scripts/render.mjs"
  echo "✅ installed to $label: $dest"
}

case "$TARGET" in
  --claude)
    install_to "$CLAUDE_DEST" "Claude Code"
    ;;
  --codex)
    install_to "$CODEX_DEST" "Codex"
    ;;
  --both)
    install_to "$CLAUDE_DEST" "Claude Code"
    install_to "$CODEX_DEST" "Codex"
    ;;
  *)
    echo "❌ usage: ./install.sh [--claude|--codex|--both]"
    exit 1
    ;;
esac

echo "🎉 done. run /$NAME with a branch, a Jira or Linear url, a bug id, or a description."
