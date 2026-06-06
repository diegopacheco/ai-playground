#!/usr/bin/env bash
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="$HOME/.claude/skills/data-dict"

rm -rf "$DEST"
mkdir -p "$DEST/scripts" "$DEST/assets"
cp "$SRC/SKILL.md" "$DEST/SKILL.md"
cp "$SRC/scripts/datadict.py" "$DEST/scripts/datadict.py"
cp "$SRC/assets/template.html" "$DEST/assets/template.html"

if [ -f "$SRC/README.md" ]; then
  cp "$SRC/README.md" "$DEST/README.md"
fi
if [ -f "$SRC/design-doc.md" ]; then
  cp "$SRC/design-doc.md" "$DEST/design-doc.md"
fi

if ! command -v python3 > /dev/null 2>&1; then
  echo "warning: python3 not found; /data-dict needs python3 to run the discovery engine"
fi

echo "installed data-dict skill to $DEST"
echo "usage: /data-dict            scan the current repository"
echo "       /data-dict sample/    scan a subdirectory"
