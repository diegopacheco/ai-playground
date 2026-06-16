#!/bin/bash
set -e
DEST="$HOME/.claude/skills/bs-claudemd"
if [ -d "$DEST" ]; then
  rm -rf "$DEST"
  echo "bs-claudemd skill removed from $DEST"
else
  echo "bs-claudemd skill is not installed"
fi
