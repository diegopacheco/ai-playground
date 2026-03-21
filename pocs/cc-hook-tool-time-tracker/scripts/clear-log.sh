#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="$HOME/.claude-hooks/tool-time-tracker.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "No log file found at $LOG_FILE"
    exit 0
fi

FILE_SIZE=$(du -h "$LOG_FILE" | cut -f1)
ENTRIES=$(wc -l < "$LOG_FILE" | tr -d ' ')

rm -f "$LOG_FILE"
echo "Cleared $ENTRIES entries ($FILE_SIZE) from $LOG_FILE"
