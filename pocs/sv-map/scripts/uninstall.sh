#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

APP_NAME="SV Map"
LSREGISTER="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"

osascript -e "tell application \"$APP_NAME\" to quit" >/dev/null 2>&1 || true
tries=10
while pgrep -f "$APP_NAME.app/Contents/MacOS" >/dev/null 2>&1 && [ "$tries" -gt 0 ]; do
  sleep 1
  tries=$((tries - 1))
done
pkill -KILL -f "$APP_NAME.app/Contents/MacOS" 2>/dev/null || true

"$SCRIPTS/stop-all.sh"

for bundle in "/Applications/$APP_NAME.app" "$HOME/Applications/$APP_NAME.app"; do
  if [ -d "$bundle" ]; then
    "$LSREGISTER" -u "$bundle" 2>/dev/null || true
    rm -rf "$bundle"
    log "removed $bundle"
  fi
done
log "$APP_NAME uninstalled"
