#!/bin/bash
TARGET="$HOME/.claude/skills/vsdd"
if [ -L "$TARGET" ] || [ -e "$TARGET" ]; then
  rm -rf "$TARGET"
  echo "VSDD skill removed from $TARGET"
else
  echo "VSDD skill not found at $TARGET"
fi
