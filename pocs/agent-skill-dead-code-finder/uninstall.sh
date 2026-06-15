#!/bin/bash
set -e
DEST="$HOME/.claude/skills/dead-code"
if [ -d "$DEST" ]; then
  rm -rf "$DEST"
  echo "dead-code skill removed from $DEST"
else
  echo "dead-code skill is not installed"
fi
