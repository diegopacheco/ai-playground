#!/usr/bin/env bash
set -euo pipefail

DEST="$HOME/.claude/skills/branch-tombs"

if [ -d "$DEST" ]; then
  rm -rf "$DEST"
  echo "branch-tombs skill removed from $DEST"
else
  echo "branch-tombs skill not found at $DEST"
fi
