#!/usr/bin/env bash
set -euo pipefail
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
BIN="monitor-stats"
TARGET="$INSTALL_DIR/$BIN"

if [ -f "$TARGET" ]; then
  rm -f "$TARGET"
  echo "removed $TARGET"
else
  echo "nothing to remove at $TARGET"
fi
