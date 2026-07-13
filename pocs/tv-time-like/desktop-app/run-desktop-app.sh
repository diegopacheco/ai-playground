#!/usr/bin/env bash
set -e
install_dir="${REELMARK_INSTALL_DIR:-/Applications}"
target="$install_dir/Reelmark.app"
if [ ! -d "$target" ]; then
  echo "Reelmark is not installed. Run ./install.sh first."
  exit 1
fi
open "$target"
