#!/usr/bin/env bash
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
SKILL_SRC="$SRC/skills/inventory"
CLAUDE_DEST="$HOME/.claude/skills/inventory"
CODEX_DEST="$HOME/.codex/skills/inventory"

echo "🔍 Codebase Inventory Skill — installer"
echo ""

if ! command -v python3 >/dev/null 2>&1; then
  echo "❌ python3 is required and was not found on PATH"
  exit 1
fi
if [ ! -f "$SKILL_SRC/SKILL.md" ]; then
  echo "❌ SKILL.md not found at $SKILL_SRC"
  exit 1
fi

TARGET="${1:-}"
if [ -z "$TARGET" ]; then
  echo "Where should the skill be installed?"
  echo "  1) 🤖 Claude Code   → $CLAUDE_DEST"
  echo "  2) 🧠 Codex         → $CODEX_DEST"
  echo "  3) 🚀 Both"
  echo ""
  printf "Pick 1, 2 or 3 [3]: "
  read -r choice < /dev/tty || choice=3
  case "${choice:-3}" in
    1) TARGET="claude" ;;
    2) TARGET="codex" ;;
    3|"") TARGET="both" ;;
    *) echo "❌ '$choice' is not 1, 2 or 3"; exit 1 ;;
  esac
fi

case "$TARGET" in
  claude|codex|both) ;;
  *) echo "❌ usage: ./install.sh [claude|codex|both]"; exit 1 ;;
esac

deploy() {
  local dest="$1" label="$2" emoji="$3"
  rm -rf "$dest"
  mkdir -p "$dest"
  cp -R "$SKILL_SRC/." "$dest/"
  find "$dest" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
  mkdir -p "$dest/prompts"
  cp "$SRC/prompts/pass-1-collect.md" "$SRC/prompts/pass-2-verify.md" \
     "$SRC/prompts/pass-3-adversarial.md" "$SRC/prompts/schema.md" "$dest/prompts/"
  if [ -f "$SRC/README.md" ]; then cp "$SRC/README.md" "$dest/README.md"; fi
  if [ -f "$SRC/design-doc.md" ]; then cp "$SRC/design-doc.md" "$dest/design-doc.md"; fi
  chmod +x "$dest/scripts/scan.py" "$dest/scripts/render.py"
  echo "$emoji Installed to $label: $dest"
}

if [ "$TARGET" = "claude" ] || [ "$TARGET" = "both" ]; then
  deploy "$CLAUDE_DEST" "Claude Code" "🤖"
fi

if [ "$TARGET" = "codex" ] || [ "$TARGET" = "both" ]; then
  if [ -d "$HOME/.codex" ]; then
    deploy "$CODEX_DEST" "Codex" "🧠"
  else
    echo "⚠️  Codex not found at $HOME/.codex — skipping Codex install"
  fi
fi

echo ""
echo "✅ Done."
echo "   /inventory              scan the current repository"
echo "   /inventory path/to/dir  scan a subdirectory"
