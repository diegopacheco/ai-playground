#!/usr/bin/env bash
set -euo pipefail

if ! command -v jq &> /dev/null; then
    echo "ERROR: jq is required but not installed."
    echo "Install it with: brew install jq"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$HOME/.claude-hooks/cc-tool-time-tracker"
SETTINGS_FILE="$HOME/.claude/settings.json"

mkdir -p "$INSTALL_DIR"
cp "$SCRIPT_DIR/hooks/track-start.sh" "$INSTALL_DIR/track-start.sh"
cp "$SCRIPT_DIR/hooks/track-end.sh" "$INSTALL_DIR/track-end.sh"
chmod +x "$INSTALL_DIR/track-start.sh"
chmod +x "$INSTALL_DIR/track-end.sh"

echo "Hooks installed to $INSTALL_DIR"

mkdir -p "$HOME/.claude"

if [ ! -f "$SETTINGS_FILE" ]; then
    echo '{}' > "$SETTINGS_FILE"
fi

cp "$SETTINGS_FILE" "${SETTINGS_FILE}.bak"
echo "Backed up settings to ${SETTINGS_FILE}.bak"

HOOK_CONFIG=$(cat <<'HOOKJSON'
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "$HOME/.claude-hooks/cc-tool-time-tracker/track-start.sh",
            "timeout": 5
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "$HOME/.claude-hooks/cc-tool-time-tracker/track-end.sh",
            "timeout": 5
          }
        ]
      }
    ]
  }
}
HOOKJSON
)

MERGED=$(jq -s '
  def merge_hooks:
    if .[0].hooks and .[1].hooks then
      .[0] * {hooks: (
        .[0].hooks as $a |
        .[1].hooks as $b |
        ($a | keys) + ($b | keys) | unique | map(
          . as $k |
          (($a[$k] // []) + ($b[$k] // []))
        ) | . as $vals |
        (($a | keys) + ($b | keys) | unique) as $keys |
        reduce range($keys | length) as $i ({}; . + {($keys[$i]): $vals[$i]})
      )}
    else
      .[0] * .[1]
    end;
  merge_hooks
' <(cat "$SETTINGS_FILE") <(echo "$HOOK_CONFIG"))

echo "$MERGED" > "$SETTINGS_FILE"

echo "Settings updated at $SETTINGS_FILE"
echo ""
echo "cc-tool-time-tracker installed successfully!"
echo "Tool call timing will be logged to ~/.claude-hooks/tool-time-tracker.log"
