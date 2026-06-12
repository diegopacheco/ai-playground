#!/bin/bash
set -e
SRC="$(cd "$(dirname "$0")" && pwd)/skill"
DEST="$HOME/.claude/skills/flamegraph"
mkdir -p "$DEST"
cp "$SRC/SKILL.md" "$DEST/SKILL.md"
cp "$SRC/build_flamegraph.py" "$DEST/build_flamegraph.py"
echo "flamegraph skill installed at $DEST"
echo "restart claude code and run /flamegraph in any java project"
