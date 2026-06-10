#!/usr/bin/env bash
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="$HOME/.claude/skills/branch-tombs"

rm -rf "$DEST"
mkdir -p "$DEST/scripts" "$DEST/assets"
cp "$SRC/SKILL.md" "$DEST/SKILL.md"
cp "$SRC/scripts/branch_tombs.py" "$DEST/scripts/branch_tombs.py"
cp "$SRC/assets/template.html" "$DEST/assets/template.html"

if [ -f "$SRC/README.md" ]; then
  cp "$SRC/README.md" "$DEST/README.md"
fi
if [ -f "$SRC/design-doc.md" ]; then
  cp "$SRC/design-doc.md" "$DEST/design-doc.md"
fi

if ! command -v python3 > /dev/null 2>&1; then
  echo "warning: python3 not found; /branch-tombs needs python3 to run the engine"
fi

echo "installed branch-tombs skill to $DEST"
echo "usage: /branch-tombs          stale after 30 days (default)"
echo "       /branch-tombs 60       stale after 60 days"
echo "       /branch-tombs owner/repo   clone a GitHub repo and bury it"
