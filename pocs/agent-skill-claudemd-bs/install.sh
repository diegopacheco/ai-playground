#!/bin/bash
set -e
SRC="$(cd "$(dirname "$0")" && pwd)/skill"
DEST="$HOME/.claude/skills/bs-claudemd"
mkdir -p "$DEST"
cp "$SRC/SKILL.md" "$DEST/SKILL.md"
cp "$SRC/bs_claudemd.py" "$DEST/bs_claudemd.py"
echo "bs-claudemd skill installed at $DEST"
echo "restart claude code and run /bs-claudemd to audit your global CLAUDE.md"
