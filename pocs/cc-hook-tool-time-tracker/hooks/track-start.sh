#!/usr/bin/env bash
set -euo pipefail

INPUT=$(cat)

TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // "unknown"' 2>/dev/null) || exit 0
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // "unknown"' 2>/dev/null) || exit 0

STATE_DIR="/tmp/cc-tool-timing"
mkdir -p "$STATE_DIR"

NOW_MS=$(perl -MTime::HiRes=time -e 'printf "%d\n", time*1000')
echo "$NOW_MS" > "${STATE_DIR}/${SESSION_ID}_${TOOL_NAME}.start"

exit 0
