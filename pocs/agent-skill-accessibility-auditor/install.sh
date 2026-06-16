#!/usr/bin/env bash
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)/skill"
CLAUDE_DEST="$HOME/.claude/skills/accessibility-auditor"
CODEX_DEST="$HOME/.codex/skills/accessibility-auditor"

if ! command -v node >/dev/null 2>&1; then
  echo "node is required"
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required"
  exit 1
fi
if [ ! -f "$SRC/SKILL.md" ]; then
  echo "ERROR: SKILL.md not found at $SRC"
  exit 1
fi

rm -rf "$CLAUDE_DEST"
mkdir -p "$CLAUDE_DEST"
cp -R "$SRC/." "$CLAUDE_DEST"
rm -rf "$CLAUDE_DEST/node_modules"

cd "$CLAUDE_DEST"
npm install --silent --no-audit --no-fund
chmod +x "$CLAUDE_DEST/scripts/audit.mjs"
echo "Installed accessibility-auditor to Claude Code: $CLAUDE_DEST"

if [ -d "$HOME/.codex" ]; then
  rm -rf "$CODEX_DEST"
  mkdir -p "$(dirname "$CODEX_DEST")"
  cp -R "$CLAUDE_DEST" "$CODEX_DEST"
  echo "Installed accessibility-auditor to Codex: $CODEX_DEST"
else
  echo "Codex not found at $HOME/.codex - skipping Codex install"
fi

echo "Done. Run /accessibility-auditor against a URL or web project."
