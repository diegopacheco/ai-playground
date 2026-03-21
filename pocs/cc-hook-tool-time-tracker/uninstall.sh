#!/usr/bin/env bash
set -euo pipefail

if ! command -v jq &> /dev/null; then
    echo "ERROR: jq is required but not installed."
    exit 1
fi

INSTALL_DIR="$HOME/.claude-hooks/cc-tool-time-tracker"
SETTINGS_FILE="$HOME/.claude/settings.json"

if [ -d "$INSTALL_DIR" ]; then
    rm -rf "$INSTALL_DIR"
    echo "Removed hooks from $INSTALL_DIR"
else
    echo "Hook directory not found at $INSTALL_DIR (already removed?)"
fi

if [ -f "$SETTINGS_FILE" ]; then
    cp "$SETTINGS_FILE" "${SETTINGS_FILE}.bak"

    UPDATED=$(jq '
      if .hooks then
        .hooks |= (
          if .PreToolUse then
            .PreToolUse |= map(select(.hooks | all(.command | contains("cc-tool-time-tracker") | not)))
            | if .PreToolUse == [] then del(.PreToolUse) else . end
          else . end
          |
          if .PostToolUse then
            .PostToolUse |= map(select(.hooks | all(.command | contains("cc-tool-time-tracker") | not)))
            | if .PostToolUse == [] then del(.PostToolUse) else . end
          else . end
        )
        | if .hooks == {} then del(.hooks) else . end
      else . end
    ' "$SETTINGS_FILE")

    echo "$UPDATED" > "$SETTINGS_FILE"
    echo "Removed hook entries from $SETTINGS_FILE"
fi

LOG_FILE="$HOME/.claude-hooks/tool-time-tracker.log"
if [ -f "$LOG_FILE" ]; then
    echo "Log file exists at $LOG_FILE"
    echo "Run: rm $LOG_FILE   (to delete it)"
fi

rm -rf /tmp/cc-tool-timing

echo ""
echo "cc-tool-time-tracker uninstalled successfully!"
