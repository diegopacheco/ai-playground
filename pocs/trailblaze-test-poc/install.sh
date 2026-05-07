#!/usr/bin/env bash
set -euo pipefail

TRAILBLAZE_BIN="$HOME/.trailblaze/bin/trailblaze"

if [ -x "$TRAILBLAZE_BIN" ]; then
  echo "trailblaze already installed at $TRAILBLAZE_BIN"
  "$TRAILBLAZE_BIN" --version || true
  exit 0
fi

echo "Installing trailblaze..."
curl -fsSL https://raw.githubusercontent.com/block/trailblaze/main/install.sh | bash

"$TRAILBLAZE_BIN" --version
