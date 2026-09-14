#!/usr/bin/env bash
set -euo pipefail

CLAUDE_DEST="$HOME/.claude/skills/skill-sec-checker"
CODEX_DEST="$HOME/.codex/skills/skill-sec-checker"
DATA_DIR="$HOME/.skill-sec-checker"

echo "🛡️  Skill Sec Checker — uninstaller"
echo ""

TARGET="${1:-}"
if [ -z "$TARGET" ]; then
  echo "What should be removed?"
  echo "  1) 🤖 Claude Code   → $CLAUDE_DEST"
  echo "  2) 🧠 Codex         → $CODEX_DEST"
  echo "  3) 🧹 Both"
  echo ""
  printf "Pick 1, 2 or 3 [3]: "
  read -r choice < /dev/tty || choice=3
  case "${choice:-3}" in
    1) TARGET="claude" ;;
    2) TARGET="codex" ;;
    3) TARGET="both" ;;
    *) echo "❌ '$choice' is not 1, 2 or 3"; exit 1 ;;
  esac
fi

case "$TARGET" in
  claude|codex|both) ;;
  *) echo "❌ usage: ./uninstall.sh [claude|codex|both]"; exit 1 ;;
esac

drop() {
  local dest="$1" label="$2"
  if [ -d "$dest" ]; then
    rm -rf "$dest"
    echo "🗑️  Removed from $label: $dest"
  else
    echo "➖ Not installed in $label — nothing to remove"
  fi
}

if [ "$TARGET" = "claude" ] || [ "$TARGET" = "both" ]; then
  drop "$CLAUDE_DEST" "Claude Code"
fi

if [ "$TARGET" = "codex" ] || [ "$TARGET" = "both" ]; then
  drop "$CODEX_DEST" "Codex"
fi

if [ -d "$DATA_DIR" ] && [ ! -d "$CLAUDE_DEST" ] && [ ! -d "$CODEX_DEST" ]; then
  printf "📜 Also delete the scan history in %s? [y/N]: " "$DATA_DIR"
  read -r wipe < /dev/tty || wipe=n
  if [ "$wipe" = "y" ] || [ "$wipe" = "Y" ]; then
    rm -rf "$DATA_DIR"
    echo "🗑️  Removed history: $DATA_DIR"
  else
    echo "📦 Kept history: $DATA_DIR"
  fi
fi

echo ""
echo "✅ Done. /skill-sec-checker uninstalled."
