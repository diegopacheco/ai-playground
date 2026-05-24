#!/usr/bin/env bash
set -euo pipefail

KEEP_DATA=0
for arg in "$@"; do
  case "$arg" in
    --keep-data) KEEP_DATA=1 ;;
    -h|--help)
      printf 'usage: %s [--keep-data]\n' "$0"; exit 0 ;;
    *) printf 'unknown flag: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

DATA_DIR="$HOME/.cc-token-bar"
SETTINGS="$HOME/.claude/settings.json"
APP_DIR="/Applications/cc-token-bar.app"
LAUNCH_AGENT="$HOME/Library/LaunchAgents/com.cc-token-bar.plist"

if [[ -f "$LAUNCH_AGENT" ]]; then
  printf 'removing LaunchAgent\n'
  launchctl unload "$LAUNCH_AGENT" 2>/dev/null || true
  rm -f "$LAUNCH_AGENT"
fi

printf 'stopping app\n'
osascript -e 'tell application "cc-token-bar" to quit' 2>/dev/null || true
pkill -f "$APP_DIR/Contents/MacOS/cc-token-bar" 2>/dev/null || true

if [[ -d "$APP_DIR" ]]; then
  printf 'removing %s\n' "$APP_DIR"
  rm -rf "$APP_DIR"
fi

if [[ -f "$SETTINGS" ]] && command -v jq >/dev/null 2>&1; then
  ts="$(date +%Y%m%d_%H%M%S)"
  cp "$SETTINGS" "$SETTINGS.bak.$ts"
  printf 'removing hook entries from %s (backup: %s.bak.%s)\n' "$SETTINGS" "$SETTINGS" "$ts"
  jq --arg prefix "$DATA_DIR/bin/hook.sh" '
    def strip_cmds:
      map(.hooks |= map(select(.command // "" | startswith($prefix) | not)))
      | map(select(.hooks | length > 0));
    if .hooks then
      .hooks |= with_entries(.value |= strip_cmds)
      | .hooks |= with_entries(select(.value | length > 0))
    else . end
  ' "$SETTINGS" > "$SETTINGS.tmp"
  mv "$SETTINGS.tmp" "$SETTINGS"
fi

if [[ "$KEEP_DATA" -eq 1 ]]; then
  printf 'keeping data at %s\n' "$DATA_DIR"
else
  if [[ -d "$DATA_DIR" ]]; then
    printf 'removing %s\n' "$DATA_DIR"
    rm -rf "$DATA_DIR"
  fi
fi

printf 'uninstall complete\n'