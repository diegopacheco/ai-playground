#!/usr/bin/env bash
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="$HOME/.claude/skills/bus-factor"

rm -rf "$DEST"
mkdir -p "$DEST/scripts" "$DEST/assets"
cp "$SRC/SKILL.md" "$DEST/SKILL.md"
cp "$SRC/scripts/bus_factor.py" "$DEST/scripts/bus_factor.py"
cp "$SRC/assets/template.html" "$DEST/assets/template.html"

if [ -f "$SRC/README.md" ]; then
  cp "$SRC/README.md" "$DEST/README.md"
fi
if [ -f "$SRC/design-doc.md" ]; then
  cp "$SRC/design-doc.md" "$DEST/design-doc.md"
fi

if ! command -v python3 > /dev/null 2>&1; then
  echo "warning: python3 not found; /bus-factor needs python3 to run the analysis engine"
fi

echo "installed bus-factor skill to $DEST"
echo "usage: /bus-factor            scan the whole repo"
echo "       /bus-factor src/       scan a subdirectory"
