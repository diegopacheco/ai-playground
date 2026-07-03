#!/bin/bash

CLAUDE_BIN=/Users/diegopacheco/.local/bin/claude
LOG_FILE=/Users/diegopacheco/claude-window-starter.log
PROMPT="hi"
MAX_WAIT=4

"$CLAUDE_BIN" -p "$PROMPT" > /dev/null 2>&1 &
PID=$!

waited=0
while [ "$waited" -lt "$MAX_WAIT" ]; do
    if ! kill -0 "$PID" 2>/dev/null; then
        break
    fi
    sleep 1
    waited=$((waited + 1))
done

if kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null
    wait "$PID" 2>/dev/null
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') window triggered" >> "$LOG_FILE"
