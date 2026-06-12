#!/bin/bash
set -e
DEST="$HOME/.claude/skills/flamegraph"
if [ -d "$DEST" ]; then
  rm -rf "$DEST"
  echo "flamegraph skill removed from $DEST"
else
  echo "flamegraph skill is not installed"
fi
