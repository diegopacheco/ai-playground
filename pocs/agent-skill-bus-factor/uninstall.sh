#!/usr/bin/env bash
set -euo pipefail

DEST="$HOME/.claude/skills/bus-factor"

if [ -d "$DEST" ]; then
  rm -rf "$DEST"
  echo "bus-factor skill removed from $DEST"
else
  echo "bus-factor skill not found at $DEST"
fi
