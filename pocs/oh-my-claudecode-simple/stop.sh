#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -f "$SCRIPT_DIR/backend.pid" ]; then
    PID=$(cat "$SCRIPT_DIR/backend.pid")
    kill $PID 2>/dev/null
    pkill -P $PID 2>/dev/null
    rm "$SCRIPT_DIR/backend.pid"
fi

if [ -f "$SCRIPT_DIR/frontend.pid" ]; then
    PID=$(cat "$SCRIPT_DIR/frontend.pid")
    kill $PID 2>/dev/null
    pkill -P $PID 2>/dev/null
    rm "$SCRIPT_DIR/frontend.pid"
fi

echo "Stopped"
