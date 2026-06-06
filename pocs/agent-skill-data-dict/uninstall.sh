#!/usr/bin/env bash
set -euo pipefail

DEST="$HOME/.claude/skills/data-dict"

if [ -d "$DEST" ]; then
  rm -rf "$DEST"
  echo "data-dict skill removed from $DEST"
else
  echo "data-dict skill not found at $DEST"
fi
