#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -f "$SCRIPT_DIR/.backend.pid" ]; then
    kill "$(cat "$SCRIPT_DIR/.backend.pid")" 2>/dev/null
    rm "$SCRIPT_DIR/.backend.pid"
    echo "Backend stopped"
fi

if [ -f "$SCRIPT_DIR/.frontend.pid" ]; then
    kill "$(cat "$SCRIPT_DIR/.frontend.pid")" 2>/dev/null
    rm "$SCRIPT_DIR/.frontend.pid"
    echo "Frontend stopped"
fi

pkill -f "agent-drawing-action-backend" 2>/dev/null
pkill -f "vite.*5173" 2>/dev/null
echo "All processes stopped"
