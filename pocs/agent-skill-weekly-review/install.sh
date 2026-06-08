#!/usr/bin/env bash
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)/weekly-review"
DEST="$HOME/.claude/skills/weekly-review"
if [ ! -d "$SRC" ]; then
  echo "source skill not found at $SRC"
  exit 1
fi
mkdir -p "$HOME/.claude/skills"
rm -rf "$DEST"
cp -R "$SRC" "$DEST"
chmod +x "$DEST/scripts/weekly_review.py"
echo "Installed weekly-review skill to $DEST"
echo "Run /weekly-review inside any git repository."
