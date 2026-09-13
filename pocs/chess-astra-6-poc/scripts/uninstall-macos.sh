#!/usr/bin/env bash
set -euo pipefail
MACOS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/macos"
APP_NAME="Wizards Gambit"
BUNDLE_ID="com.diegopacheco.wizardsgambit"

running() { pgrep -x "$APP_NAME" >/dev/null 2>&1; }

if running; then
  printf 'Quitting %s (this stops the game server)...\n' "$APP_NAME"
  osascript -e "tell application id \"$BUNDLE_ID\" to quit" >/dev/null 2>&1 || true
  for ((attempt=0; attempt<30; attempt++)); do
    running || break
    sleep 1
  done
  if running; then
    pkill -x "$APP_NAME" || true
    sleep 1
  fi
fi

installed=("/Applications/$APP_NAME.app" "$HOME/Applications/$APP_NAME.app")
while IFS= read -r found; do
  [ -n "$found" ] && installed+=("$found")
done < <(mdfind "kMDItemCFBundleIdentifier == '$BUNDLE_ID'" 2>/dev/null || true)

for bundle in "${installed[@]}"; do
  if [ -d "$bundle" ]; then
    rm -rf "$bundle"
    printf 'Removed %s\n' "$bundle"
  fi
done
rm -rf "$MACOS/build" "$MACOS/app/config.json" "$MACOS/app/logo.svg"
printf '%s uninstalled.\n' "$APP_NAME"
