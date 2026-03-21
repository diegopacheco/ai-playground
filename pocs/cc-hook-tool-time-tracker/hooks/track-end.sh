#!/usr/bin/env bash
set -euo pipefail

INPUT=$(cat)

TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // "unknown"' 2>/dev/null) || exit 0
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // "unknown"' 2>/dev/null) || exit 0
CWD=$(echo "$INPUT" | jq -r '.cwd // "unknown"' 2>/dev/null) || exit 0

STATE_DIR="/tmp/cc-tool-timing"
STATE_FILE="${STATE_DIR}/${SESSION_ID}_${TOOL_NAME}.start"

if [ ! -f "$STATE_FILE" ]; then
    exit 0
fi

START_MS=$(cat "$STATE_FILE")
NOW_MS=$(perl -MTime::HiRes=time -e 'printf "%d\n", time*1000')
ELAPSED_MS=$((NOW_MS - START_MS))

LOG_DIR="$HOME/.claude-hooks"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/tool-time-tracker.log"

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "{\"timestamp\":\"${TIMESTAMP}\",\"session_id\":\"${SESSION_ID}\",\"tool\":\"${TOOL_NAME}\",\"elapsed_ms\":${ELAPSED_MS},\"cwd\":\"${CWD}\"}" >> "$LOG_FILE"

rm -f "$STATE_FILE"

exit 0
