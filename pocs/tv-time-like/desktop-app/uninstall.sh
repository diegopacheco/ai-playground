#!/usr/bin/env bash
set -e
install_dir="${REELMARK_INSTALL_DIR:-/Applications}"
target="$install_dir/Reelmark.app"
osascript -e 'tell application id "com.diegopacheco.reelmark" to quit' 2>/dev/null || true
if [ -w "$install_dir" ]; then
  rm -rf "$target"
else
  sudo rm -rf "$target"
fi
echo "Uninstalled $target"
