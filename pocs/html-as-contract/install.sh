#!/usr/bin/env bash
set -euo pipefail
SRC="$(cd "$(dirname "$0")/skill" && pwd)"
DEST="$HOME/.claude/skills/html-as-contract"
mkdir -p "$DEST"
cp "$SRC/SKILL.md" "$DEST/SKILL.md"
cp "$SRC/template.html" "$DEST/template.html"
echo "installed html-as-contract to $DEST"
