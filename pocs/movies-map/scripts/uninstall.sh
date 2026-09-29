#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

APP_NAME="Movies Map"
BUNDLE_ID="com.diegopacheco.moviesmap"
INSTALL_DIR="${MOVIES_MAP_INSTALL_DIR:-/Applications}"

osascript -e "tell application id \"$BUNDLE_ID\" to quit" >/dev/null 2>&1 || true
tries=15
while pgrep -f "$APP_NAME.app/Contents/MacOS" >/dev/null 2>&1 && [ "$tries" -gt 0 ]; do
  sleep 1
  tries=$((tries - 1))
done
pkill -f "$APP_NAME.app/Contents/MacOS" 2>/dev/null || true
for app in "$INSTALL_DIR/$APP_NAME.app" "$HOME/Applications/$APP_NAME.app"; do
  if [ -d "$app" ]; then
    rm -rf "$app"
    log "removed $app"
  fi
done
log "uninstalled $APP_NAME"
