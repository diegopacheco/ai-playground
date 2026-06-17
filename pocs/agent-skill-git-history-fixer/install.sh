#!/usr/bin/env bash
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)/skill"
DEST="$HOME/.claude/skills/fix-git-history"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required"
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "git is required"
  exit 1
fi
if [ ! -f "$SRC/SKILL.md" ]; then
  echo "ERROR: SKILL.md not found at $SRC"
  exit 1
fi

rm -rf "$DEST"
mkdir -p "$DEST"
cp "$SRC/SKILL.md" "$DEST/SKILL.md"
cp "$SRC/fix_git_history.py" "$DEST/fix_git_history.py"
chmod +x "$DEST/fix_git_history.py"

echo "fix-git-history installed at $DEST"
echo "restart claude code and run /fix-git-history in any git repository"
