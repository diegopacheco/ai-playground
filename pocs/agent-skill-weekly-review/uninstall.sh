#!/usr/bin/env bash
set -euo pipefail
DEST="$HOME/.claude/skills/weekly-review"
if [ -d "$DEST" ]; then
  rm -rf "$DEST"
  echo "Removed weekly-review skill from $DEST"
else
  echo "weekly-review skill not installed at $DEST"
fi
