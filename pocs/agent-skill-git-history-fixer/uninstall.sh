#!/usr/bin/env bash
set -euo pipefail

DEST="$HOME/.claude/skills/fix-git-history"

if [ -d "$DEST" ]; then
  rm -rf "$DEST"
  echo "fix-git-history removed from $DEST"
else
  echo "fix-git-history is not installed"
fi
