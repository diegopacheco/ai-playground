#!/bin/bash
set -e
SRC="$(cd "$(dirname "$0")" && pwd)/skill"
DEST="$HOME/.claude/skills/dead-code"
mkdir -p "$DEST"
cp "$SRC/SKILL.md" "$DEST/SKILL.md"
cp "$SRC/find_dead_code.py" "$DEST/find_dead_code.py"
echo "dead-code skill installed at $DEST"
echo "restart claude code and run /dead-code in any java project"
