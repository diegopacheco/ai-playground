#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
BIN="monitor-stats"

cargo build --release
mkdir -p "$INSTALL_DIR"
install -m 0755 "target/release/$BIN" "$INSTALL_DIR/$BIN"
echo "installed $INSTALL_DIR/$BIN"

case ":$PATH:" in
  *":$INSTALL_DIR:"*) ;;
  *) echo "warning: $INSTALL_DIR is not on PATH, add: export PATH=\"$INSTALL_DIR:\$PATH\"" ;;
esac
