#!/usr/bin/env bash
set -euo pipefail
DEST="$HOME/.claude/skills/prometheus-runbooks"
if [ -d "$DEST" ]; then
  rm -rf "$DEST"
  echo "Removed $DEST"
else
  echo "Nothing to remove at $DEST"
fi
